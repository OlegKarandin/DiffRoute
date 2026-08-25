"""AllocationHead: the per-(demand, boundary) allocation contract.

Spec section 8 items 2-5. The load-bearing one is
test_greedy_weights_reproduce_the_oracle: it proves a positive oracle gap
in a real run is an OPTIMISATION failure, not an architectural one.
"""
import math

import pytest
import torch

from diffopt.placement.allocation import (
    ALLOC_FEATURE_DIM,
    LOOKAHEAD_COLS,
    ROUTE_CONTEXT_COLS,
    AllocationHead,
    site_view,
    total_device_cost,
)
from diffopt.qot.segment_combiner import GSNR_MAX, GSNR_MIN, db_to_linear_noise


def noise(db):
    """Linear noise for a segment GSNR in dB, clamped the way the head does."""
    return 10.0 ** (-min(max(db, GSNR_MIN), GSNR_MAX) / 10.0)


def test_init_is_closed_and_deterministic():
    """Spec 2.2: the head starts CLOSED at a ~ sigmoid(-3), regardless of
    features, so lambda_dev can be live from epoch 0 without a warm-up."""
    head = AllocationHead()
    feats = torch.randn(64, ALLOC_FEATURE_DIM) * 10.0
    a = torch.sigmoid(head.score(feats))
    assert torch.allclose(a, torch.full_like(a, 1.0 / (1.0 + math.exp(3.0))), atol=1e-6)


def test_final_layer_weights_are_exactly_zero_at_init():
    head = AllocationHead()
    assert torch.equal(head.net[-1].weight, torch.zeros_like(head.net[-1].weight))
    assert torch.allclose(head.net[-1].bias, torch.tensor([-3.0]))


def test_carry_under_hard_decisions_is_the_true_chunk_noise():
    """Spec section 8 item 3. Two segments, a hard cut between them: each
    chunk's noise is its own segment's, so the worst chunk is the noisier."""
    head = AllocationHead()
    with torch.no_grad():
        head.net[-1].bias.fill_(50.0)          # score >> 0 -> always cut
    seg_db = torch.tensor([[12.0, 9.0, 20.0]])
    seg_noise = db_to_linear_noise(seg_db)
    seg_km = torch.tensor([[300.0, 400.0, 500.0]])
    bar = torch.tensor([9.5])
    nseg = torch.tensor([3])

    a, a_phys = head.rollout(seg_noise, seg_km, bar, nseg, hard=True)
    assert torch.equal(a, torch.ones(1, 2))
    assert torch.equal(a, a_phys)
    # every boundary cut -> every chunk is a single segment
    worst = max(noise(12.0), noise(9.0), noise(20.0))
    assert head.last_max_chunk_noise.item() == pytest.approx(worst, rel=1e-5)


def test_no_cut_accumulates_the_whole_path():
    head = AllocationHead()                      # init: closed, but not hard-0
    with torch.no_grad():
        head.net[-1].bias.fill_(-50.0)           # score << 0 -> never cut
    seg_db = torch.tensor([[12.0, 9.0, 20.0]])
    seg_noise = db_to_linear_noise(seg_db)
    a, _ = head.rollout(
        seg_noise, torch.ones(1, 3), torch.tensor([9.5]), torch.tensor([3]), hard=True
    )
    assert torch.equal(a, torch.zeros(1, 2))
    total = noise(12.0) + noise(9.0) + noise(20.0)
    assert head.last_max_chunk_noise.item() == pytest.approx(total, rel=1e-5)


@pytest.mark.parametrize("k", [1, 2, 8])
def test_padding_and_masking_at_the_edges(k):
    """Spec section 8 item 4: K_d = 0 boundaries (one segment), 1, and J_max.
    No NaN anywhere, and no cut is ever placed past a demand's real length."""
    j = 8
    d = 4
    seg_noise = torch.rand(d, j) + 0.01
    seg_km = torch.rand(d, j) * 500.0
    bar = torch.full((d,), 9.5)
    nseg = torch.full((d,), k, dtype=torch.long)

    head = AllocationHead()
    with torch.no_grad():
        head.net[-1].bias.fill_(50.0)            # always cut where allowed
    a, _ = head.rollout(seg_noise, seg_km, bar, nseg, hard=True)

    assert torch.isfinite(a).all()
    assert torch.equal(a[:, : max(k - 1, 0)], torch.ones(d, max(k - 1, 0)))
    assert torch.equal(a[:, k - 1 :], torch.zeros(d, j - 1 - (k - 1)))


def test_single_segment_demand_has_no_decisions_and_no_nan():
    """K_d = 0: one segment, zero boundaries. The trailing chunk is not a
    decision (spec 2.1), so a is all-zero and nothing divides by zero."""
    head = AllocationHead()
    a, _ = head.rollout(
        torch.tensor([[0.5, 0.0, 0.0]]),
        torch.tensor([[100.0, 0.0, 0.0]]),
        torch.tensor([9.5]),
        torch.tensor([1]),
        hard=True,
    )
    assert torch.equal(a, torch.zeros(1, 2))
    assert torch.isfinite(head.last_max_chunk_noise).all()


def test_lookahead_off_zeroes_only_the_two_lookahead_columns():
    on = AllocationHead(lookahead=True)
    off = AllocationHead(lookahead=False)
    off.load_state_dict(on.state_dict())
    with torch.no_grad():                        # break the zero-init degeneracy
        on.net[-1].weight.normal_()
        off.net[-1].weight.copy_(on.net[-1].weight)

    feats = torch.randn(16, ALLOC_FEATURE_DIM)
    masked = feats.clone()
    masked[:, [3, 4]] = 0.0
    assert torch.allclose(off.score(feats), on.score(masked), atol=1e-6)
    assert not torch.allclose(off.score(feats), on.score(feats), atol=1e-4)


def test_route_context_mask_zeroes_features_5_6_7():
    on = AllocationHead(route_context=True)
    off = AllocationHead(route_context=False)
    off.load_state_dict(on.state_dict())
    with torch.no_grad():                        # break the zero-init degeneracy
        on.net[-1].weight.normal_()
        off.net[-1].weight.copy_(on.net[-1].weight)

    feats = torch.randn(16, ALLOC_FEATURE_DIM)
    masked = feats.clone()
    masked[:, list(ROUTE_CONTEXT_COLS)] = 0.0
    assert torch.allclose(off.score(feats), on.score(masked), atol=1e-6)
    assert not torch.allclose(off.score(feats), on.score(feats), atol=1e-4)


def test_lookahead_and_route_context_off_together_zero_all_five_columns():
    on = AllocationHead(lookahead=True, route_context=True)
    off = AllocationHead(lookahead=False, route_context=False)
    off.load_state_dict(on.state_dict())
    with torch.no_grad():                        # break the zero-init degeneracy
        on.net[-1].weight.normal_()
        off.net[-1].weight.copy_(on.net[-1].weight)

    feats = torch.randn(16, ALLOC_FEATURE_DIM)
    masked = feats.clone()
    masked[:, list(LOOKAHEAD_COLS) + list(ROUTE_CONTEXT_COLS)] = 0.0
    assert torch.allclose(off.score(feats), on.score(masked), atol=1e-6)
    assert not torch.allclose(off.score(feats), on.score(feats), atol=1e-4)


def test_dropout_splits_priced_from_physics():
    """alloc_dropout zeroes the PHYSICS decision (manufacturing a violation
    so the hinge reopens) while the PRICED decision stays undropped — else
    the price per device fluctuates with the mask."""
    head = AllocationHead()
    head.train()
    with torch.no_grad():
        head.net[-1].bias.fill_(50.0)
    a, a_phys = head.rollout(
        torch.rand(32, 5) + 0.01, torch.rand(32, 5) * 100,
        torch.full((32,), 9.5), torch.full((32,), 5, dtype=torch.long),
        dropout_p=1.0,
    )
    assert torch.equal(a_phys, torch.zeros(32, 4))
    assert (a > 0.99).all()


def test_dropout_is_inactive_in_eval_mode():
    head = AllocationHead()
    head.eval()
    a, a_phys = head.rollout(
        torch.rand(8, 4) + 0.01, torch.rand(8, 4) * 100,
        torch.full((8,), 9.5), torch.full((8,), 4, dtype=torch.long),
        dropout_p=1.0,
    )
    assert torch.equal(a, a_phys)


def test_total_device_cost_is_sum_over_nodes_then_demands():
    """Spec section 6 hook 2: cost aggregation lives in ONE named function,
    written sum_n sum_d, so Stage IV can slot max_s between the two sums."""
    alloc = torch.tensor([[0.0, 1.0, 0.5], [1.0, 1.0, 0.0]])
    assert total_device_cost(alloc).item() == pytest.approx(3.5)


def test_site_view_is_the_per_node_max_and_never_priced():
    alloc = torch.tensor([[0.0, 1.0, 0.5], [1.0, 1.0, 0.0]])
    assert torch.equal(site_view(alloc), torch.tensor([1.0, 1.0, 0.5]))


def test_rollout_is_differentiable_in_seg_noise():
    """The device term must be route-differentiable: d(devices)/d(n) != 0.
    Spec section 8 item 5's precondition — the pipeline-level version of this
    test lives in tests/test_pipeline.py."""
    head = AllocationHead()
    with torch.no_grad():
        head.net[-1].weight.normal_(std=0.5)
    seg_noise = (torch.rand(3, 5) + 0.01).requires_grad_(True)
    a, _ = head.rollout(
        seg_noise, torch.rand(3, 5) * 100,
        torch.full((3,), 9.5), torch.full((3,), 5, dtype=torch.long),
    )
    total_device_cost(a).backward()
    assert seg_noise.grad is not None
    assert seg_noise.grad.abs().sum().item() > 0


def test_km_since_cut_resets_on_a_physics_cut():
    """Feature 5 is documented as `km_since_cut / KM_SCALE` (module docstring),
    but the accumulator was never reset on a cut -- it just tracked km_done,
    identical to what feature 6 already computes. Isolate feature 5 alone via
    the identity-first-layers trick from
    test_oracle.py::test_greedy_weights_reproduce_the_oracle, so a threshold
    on it alone drives the cut decision, then pick two segment lengths where
    the SECOND decision only cuts if the first cut failed to reset the
    accumulator: 200km cuts (0.2 > 0.15 threshold); a correctly-reset
    accumulator then reads 100km (0.1) at the next boundary and does not cut,
    while a never-reset one reads 300km (0.3) and cuts again.
    """
    head = AllocationHead()
    with torch.no_grad():
        w = torch.zeros(1, head.net[-1].weight.shape[1])
        w[0, 5] = 1.0e6
        head.net[-1].weight.copy_(w)
        head.net[0].weight.zero_()
        head.net[0].weight[:ALLOC_FEATURE_DIM, :].copy_(torch.eye(ALLOC_FEATURE_DIM))
        head.net[0].bias.fill_(50.0)       # keep ReLU in its linear region
        head.net[2].weight.zero_()
        head.net[2].weight.copy_(torch.eye(head.net[2].weight.shape[0]))
        head.net[2].bias.zero_()
        # Undo the +50 offset that carried through both identity layers, then
        # place the threshold at km_since/KM_SCALE == 0.15.
        offset = (-w @ torch.full((w.shape[1], 1), 50.0)).squeeze()
        head.net[-1].bias.copy_(offset - 0.15 * w[0, 5])

    seg_noise = torch.full((1, 3), 0.01)
    seg_km = torch.tensor([[200.0, 100.0, 50.0]])
    bar = torch.tensor([9.5])
    nseg = torch.tensor([3])

    a, a_phys = head.rollout(seg_noise, seg_km, bar, nseg, hard=True)
    assert torch.equal(a, torch.tensor([[1.0, 0.0]]))
    assert torch.equal(a, a_phys)


def test_first_segment_has_no_device_gradient_because_the_carry_is_detached():
    """Pins the carry detach (invariants.md, "regen helps is a property of the
    FOLD"). seg_noise column 0 is the ONE column whose only route into any
    score is through the carry: at k=0 the features read c (= n[:,0]) and
    n_next (= n[:,1]), and column 0 never appears as an n_next or a seg_km
    anywhere. So with `cd = c.detach()` its device-count gradient is EXACTLY
    zero, while every later column keeps a live path.

    This is the discriminator the sum-over-everything assertion in
    test_rollout_is_differentiable_in_seg_noise cannot make: reverting the fix
    to `cd = c` leaves that test — and the whole 306-test suite — green, but
    turns this column nonzero.
    """
    torch.manual_seed(0)
    head = AllocationHead()
    with torch.no_grad():
        head.net[-1].weight.normal_(std=0.5)
    seg_noise = (torch.rand(3, 5) + 0.01).requires_grad_(True)
    a, _ = head.rollout(
        seg_noise, torch.rand(3, 5) * 100,
        torch.full((3,), 9.5), torch.full((3,), 5, dtype=torch.long),
    )
    total_device_cost(a).backward()

    assert seg_noise.grad is not None
    # The carry-only column: exactly 0, not merely small.
    assert seg_noise.grad[:, 0].abs().sum().item() == 0.0
    # ...and the fix must not have killed route-differentiability outright:
    # the later columns still reach the score via g_next / g_after's n_next.
    assert seg_noise.grad[:, 1:].abs().sum().item() > 0
