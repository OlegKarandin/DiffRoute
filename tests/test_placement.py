"""RegenPlacement: the deployed-placement and count-penalty contracts."""
import math

import pytest
import torch

from diffopt.placement.regenerator import RegenPlacement


def test_hard_placement_mask_is_the_positive_logits():
    """The deployed placement is sigmoid(logit/tau) > 0.5, which for any
    tau > 0 is exactly logit > 0. Defining it on the logit directly makes it
    tau-invariant by construction rather than by argument."""
    placement = RegenPlacement(num_nodes=5)
    with torch.no_grad():
        placement.regen_logits.copy_(torch.tensor([-1.0, 0.0, 0.5, -0.01, 3.0]))

    mask = placement.hard_placement_mask()

    assert mask.dtype == torch.bool
    assert mask.tolist() == [False, False, True, False, True]


def test_hard_placement_mask_is_tau_invariant():
    placement = RegenPlacement(num_nodes=4)
    with torch.no_grad():
        placement.regen_logits.copy_(torch.tensor([-2.0, -0.1, 0.1, 2.0]))

    mask = placement.hard_placement_mask()
    for tau in (0.1, 1.0, 10.0):
        assert torch.equal((placement.get_regen_probs(tau) > 0.5), mask)


def test_hard_placement_mask_is_detached():
    """Callers threshold it and forward on it; it must never drag the
    placement head into an evaluation-only graph."""
    placement = RegenPlacement(num_nodes=3)
    assert not placement.hard_placement_mask().requires_grad


def test_zero_logits_place_nothing():
    """The neutral init sits at sigmoid(0) = 0.5 exactly, which is NOT
    strictly greater than 0.5 — so the initial deployed placement is empty.
    This is the state that produced the (violated=0, regens=0) checkpoint
    artifact; the mask reports it honestly as zero regenerators."""
    assert RegenPlacement(num_nodes=7).hard_placement_mask().sum().item() == 0


def test_count_penalty_matches_the_probability_mass_for_the_sigmoid_gate():
    placement = RegenPlacement(num_nodes=6)
    with torch.no_grad():
        placement.regen_logits.copy_(torch.tensor([-1.0, 0.0, 1.0, 2.0, -3.0, 0.5]))

    for tau in (0.1, 1.0):
        expected = placement.get_regen_probs(tau).sum()
        assert torch.allclose(placement.count_penalty(tau), expected)


def test_count_penalty_carries_gradient():
    placement = RegenPlacement(num_nodes=4)
    placement.count_penalty(1.0).backward()
    assert placement.regen_logits.grad is not None
    assert torch.all(placement.regen_logits.grad > 0)


def test_hard_concrete_gate_is_deterministic_in_eval_mode():
    placement = RegenPlacement(4, gate="hard_concrete")
    placement.eval()
    a = placement.get_regen_probs()
    b = placement.get_regen_probs()
    assert torch.equal(a, b)


def test_hard_concrete_gate_is_stochastic_in_training_mode():
    placement = RegenPlacement(64, gate="hard_concrete")
    placement.train()
    torch.manual_seed(0)
    a = placement.get_regen_probs()
    b = placement.get_regen_probs()
    assert not torch.equal(a, b)


def test_hard_concrete_gate_produces_exact_zeros_and_ones():
    """Stretch-then-clamp is the whole point: the gate can be exactly closed
    or exactly open, which a sigmoid can never be."""
    placement = RegenPlacement(2000, gate="hard_concrete")
    placement.train()
    torch.manual_seed(0)
    z = placement.get_regen_probs()
    assert (z == 0.0).any()
    assert (z == 1.0).any()
    assert torch.all((z >= 0.0) & (z <= 1.0))


def test_hard_concrete_placement_threshold():
    """z > 0 iff sigmoid(log_alpha/beta) > -gamma/(zeta-gamma). With the
    defaults that boundary is 0.1/1.2 = 0.08333, i.e. log_alpha/beta above
    logit(0.08333) = -2.3979."""
    beta, gamma, zeta = 0.5, -0.1, 1.1
    boundary = beta * math.log((-gamma / (zeta - gamma)) / (1 + gamma / (zeta - gamma)))

    placement = RegenPlacement(3, gate="hard_concrete", beta=beta, gamma=gamma, zeta=zeta)
    with torch.no_grad():
        placement.log_alpha.copy_(
            torch.tensor([boundary - 0.5, boundary + 0.5, 0.0])
        )

    assert placement.hard_placement_mask().tolist() == [False, True, True]


def test_hard_concrete_count_penalty_matches_the_closed_form():
    """The penalty is P(gate open) — the expected COUNT — not sum(p)."""
    beta, gamma, zeta = 0.5, -0.1, 1.1
    placement = RegenPlacement(3, gate="hard_concrete", beta=beta, gamma=gamma, zeta=zeta)
    with torch.no_grad():
        placement.log_alpha.copy_(torch.tensor([-1.0, 0.0, 2.0]))

    shift = beta * math.log(-gamma / zeta)
    expected = sum(1.0 / (1.0 + math.exp(-(a - shift))) for a in (-1.0, 0.0, 2.0))

    assert placement.count_penalty().item() == pytest.approx(expected, rel=1e-6)


def test_hard_concrete_count_penalty_is_not_the_probability_mass():
    """Regression guard: if someone 'simplifies' count_penalty back to
    get_regen_probs().sum(), L0's whole reason for existing is gone."""
    placement = RegenPlacement(20, gate="hard_concrete")
    placement.eval()
    assert not torch.isclose(
        placement.count_penalty(), placement.get_regen_probs().sum()
    )


def test_unknown_gate_raises_a_named_error():
    with pytest.raises(ValueError, match="gate"):
        RegenPlacement(4, gate="gumbel")


def test_sigmoid_gate_is_the_default_and_unchanged():
    """Every committed checkpoint and config depends on this."""
    placement = RegenPlacement(5)
    assert hasattr(placement, "regen_logits")
    assert not hasattr(placement, "log_alpha")
    assert torch.allclose(placement.get_regen_probs(1.0), torch.full((5,), 0.5))
