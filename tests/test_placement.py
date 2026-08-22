"""RegenPlacement: the deployed-placement and count-penalty contracts."""
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
