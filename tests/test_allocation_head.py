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
from diffopt.placement.oracle import oracle_allocation
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

    a, a_phys, _ = head.rollout(seg_noise, seg_km, bar, nseg, hard=True)
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
    a, _, _ = head.rollout(
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
    a, _, _ = head.rollout(seg_noise, seg_km, bar, nseg, hard=True)

    assert torch.isfinite(a).all()
    assert torch.equal(a[:, : max(k - 1, 0)], torch.ones(d, max(k - 1, 0)))
    assert torch.equal(a[:, k - 1 :], torch.zeros(d, j - 1 - (k - 1)))


def test_single_segment_demand_has_no_decisions_and_no_nan():
    """K_d = 0: one segment, zero boundaries. The trailing chunk is not a
    decision (spec 2.1), so a is all-zero and nothing divides by zero."""
    head = AllocationHead()
    a, _, _ = head.rollout(
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
    a, a_phys, _ = head.rollout(
        torch.rand(32, 5) + 0.01, torch.rand(32, 5) * 100,
        torch.full((32,), 9.5), torch.full((32,), 5, dtype=torch.long),
        dropout_p=1.0,
    )
    assert torch.equal(a_phys, torch.zeros(32, 4))
    assert (a > 0.99).all()


def test_dropout_is_inactive_in_eval_mode():
    head = AllocationHead()
    head.eval()
    a, a_phys, _ = head.rollout(
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
    a, _, _ = head.rollout(
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

    a, a_phys, _ = head.rollout(seg_noise, seg_km, bar, nseg, hard=True)
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
    a, _, _ = head.rollout(
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


def test_greedy_residual_init_reproduces_the_oracle():
    """The arm-3 keystone (spec 5.2). Untrained, `AllocationHead(greedy_residual=True)`
    has MLP == 0 exactly, so score = alpha * (bar_d - g_after) = -alpha *
    feature4 with alpha > 0 — i.e. score > 0 <=> feature4 < 0, which is
    exactly the oracle's own greedy cut rule
    (test_oracle.py::test_greedy_weights_reproduce_the_oracle hand-sets the
    same rule with a hand-built weight matrix). A hard=True rollout at init
    must therefore reproduce oracle_allocation bit for bit, with no training
    step in between.

    Built per plan-docs/.../task-5-brief.md "deviation 3": the spec's own
    text asks for an absolute tie guard, but the head/oracle disagreement
    was measured to begin at a RELATIVE gap of ~1e-7 — 1e-6 absolute is safe
    at bar 12.5 dB but inside the failure band at bar -4 dB. So the batch is
    reject-sampled against a relative gap guard (1e-5) walked along the
    oracle's own float64 accumulation, and seg_db is clamped to
    [GSNR_MIN, GSNR_MAX] before conversion to noise — a bare `rollout()`
    call, unlike the pipeline, does not clamp for you, and skipping that
    step here would measure a clamp-band artifact rather than head/oracle
    agreement.
    """

    def relative_tie_free(seg_db_clamped, bar_db, num_segments):
        """Replicates oracle_allocation's own float64 accumulation
        boundary-by-boundary and reports False the first time a boundary's
        gap to the bar, relative to the bar, is not clearly resolved."""
        d, j = seg_db_clamped.shape
        n = db_to_linear_noise(seg_db_clamped.double())
        bar_noise = db_to_linear_noise(bar_db.double())
        c = torch.zeros(d, dtype=torch.float64)
        for k in range(j):
            valid = k < num_segments
            n_k = n[:, k]
            if k == 0:
                c = torch.where(valid, n_k, c)
                continue
            candidate = c + n_k
            rel_gap = (candidate - bar_noise).abs() / bar_noise
            if bool((valid & (rel_gap <= 1e-5)).any()):
                return False
            over = valid & (candidate > bar_noise)
            c = torch.where(valid, torch.where(over, n_k, candidate), c)
        return True

    def draw(seed):
        g = torch.Generator().manual_seed(seed)
        # Spans the clamp band on both ends ([-10, 40] against
        # GSNR_MIN/MAX == -5/35), ragged num_segments (including K_d == 0,
        # zero boundaries), and an extreme-ish bar range -- the hostile
        # properties verified by hand during design.
        seg_db = torch.rand(D, J, generator=g) * 50.0 - 10.0
        bar_db = torch.rand(D, generator=g) * 20.0 - 4.0
        num_segments = torch.randint(1, J + 1, (D,), generator=g)
        seg_km = torch.rand(D, J, generator=g) * 400.0 + 1.0
        return seg_db, bar_db, num_segments, seg_km

    D, J = 40, 9
    seed = 20260825
    for attempt in range(200):
        seg_db, bar_db, num_segments, seg_km = draw(seed)
        seg_db_clamped = seg_db.clamp(GSNR_MIN, GSNR_MAX)
        if relative_tie_free(seg_db_clamped, bar_db, num_segments):
            break
        seed += 1
    else:
        raise AssertionError(
            "could not find a tie-free batch in 200 attempts -- widen the "
            "draw or check relative_tie_free for a bug"
        )

    head = AllocationHead(greedy_residual=True)
    seg_noise = db_to_linear_noise(seg_db_clamped)
    a_head, _, _ = head.rollout(seg_noise, seg_km, bar_db, num_segments, hard=True)
    expected = oracle_allocation(seg_db_clamped, bar_db, num_segments)

    assert torch.equal(a_head, expected.a)
    # A second, coarser assertion so a failure names the delta directly
    # rather than just "tensors not equal".
    assert int(a_head.sum().item()) == int(expected.count.sum().item())


def test_greedy_residual_rejects_lookahead_false():
    """greedy_residual reads feature 4, which lookahead=False masks to zero
    -- the combination is nonsensical, not merely unhelpful, so it raises
    rather than silently degrading to alpha * 0."""
    with pytest.raises(ValueError):
        AllocationHead(greedy_residual=True, lookahead=False)


def test_alpha_stays_positive():
    """Spec 5.2: alpha = softplus(alpha_raw) keeps alpha > 0 for any
    alpha_raw, so the sign test `score > 0 <=> feature4 < 0` never inverts
    no matter how training moves alpha_raw."""
    head = AllocationHead(greedy_residual=True)
    for v in (-50.0, 0.0, 50.0):
        with torch.no_grad():
            head.alpha_raw.fill_(v)
        assert head.alpha > 0


# ---------------------------------------------------------------------------
# Waste surcharge (arm 4, spec 5.3) — task-6-brief.md section 9, tests 5-7
# ---------------------------------------------------------------------------

def test_waste_is_zero_when_every_cut_is_justified():
    """§9 test 5. Hand-built case where every cut sits at feature4 <= 0
    (i.e. every cut is genuinely needed to stay feasible): waste must be
    exactly 0, since `relu(feature4) == 0` at every boundary that fires.

    Reuses test_carry_under_hard_decisions_is_the_true_chunk_noise's own
    construction (forced cuts everywhere via bias=50.0, seg_db=[12, 9, 20],
    bar=9.5): hand-verified below that feature4 is negative at BOTH real
    boundaries under that same input (-2.264 and -0.832 dB), so this is not
    a new hand-built case, it is the existing one re-read for a property it
    already happens to satisfy.
    """
    head = AllocationHead()
    with torch.no_grad():
        head.net[-1].bias.fill_(50.0)          # score >> 0 -> always cut
    seg_db = torch.tensor([[12.0, 9.0, 20.0]])
    seg_noise = db_to_linear_noise(seg_db)
    seg_km = torch.tensor([[300.0, 400.0, 500.0]])
    bar = torch.tensor([9.5])
    nseg = torch.tensor([3])

    a, a_phys, waste = head.rollout(seg_noise, seg_km, bar, nseg, hard=True)
    assert torch.equal(a, torch.ones(1, 2))    # both boundaries did cut
    assert waste.item() == 0.0                 # both cuts were justified


def test_waste_equals_hand_computed_surcharge():
    """§9 test 6. A hand-built (seg_noise, bar) pair whose waste value is
    computed independently (float64 `math.log10`, not the module under
    test) and asserted against.

    JUDGMENT CALL (see task-6-report.md): the plan's own trajectory
    (f4 = [+0.096, -0.209, +7.500, +5.739, +7.958]) cannot be reproduced
    without knowing the exact seg_db draw the plan's author used, which is
    not recoverable from the brief text alone -- reconstructing it would
    require guessing a specific random seed or hand-tuned dB sequence. This
    test instead uses a SELF-CONSISTENT hand-derived construction with the
    same mechanism: reuses test_km_since_cut_resets_on_a_physics_cut's exact
    weight surgery (identity first two layers, w[0, 5] = 1e6, threshold at
    km_since/KM_SCALE == 0.15) so the cut PATTERN is driven entirely by
    feature 5 and is provably independent of feature 4's value, then picks
    seg_noise so that boundary 1 (which cuts) has feature4 < 0 (justified,
    contributes 0) and boundary 3 (which also cuts) has feature4 > 0
    (wasteful, contributes its own value) -- the same qualitative shape the
    plan's own trajectory has, verified against a hand computation using
    `math.log10` at full float64 precision, independent of this module.
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
        offset = (-w @ torch.full((w.shape[1], 1), 50.0)).squeeze()
        head.net[-1].bias.copy_(offset - 0.15 * w[0, 5])

    # Hand-picked so the km_since surgery (independent of these values --
    # only feature 5, i.e. seg_km, drives the decision) produces cuts at
    # boundaries {1, 3}: a = [[0, 1, 0, 1, 0]]. Verified by hand (float64
    # math.log10) that feature4[1] = -5.5206 (justified, cuts anyway because
    # km_since forces it) and feature4[3] = +3.3400 (wasteful): waste should
    # equal exactly feature4[3]'s relu, i.e. ~3.3400.
    seg_noise = torch.tensor([[0.05, 0.30, 0.05, 0.001, 0.001, 0.01]])
    seg_km = torch.full((1, 6), 100.0)
    bar = torch.tensor([9.5])
    nseg = torch.tensor([6])

    a, a_phys, waste = head.rollout(seg_noise, seg_km, bar, nseg, hard=True)
    assert torch.equal(a, torch.tensor([[0.0, 1.0, 0.0, 1.0, 0.0]]))
    # Hand-computed via float64 math.log10, independently of this module:
    #   f4[1] = -10*log10(0.35+0.05) - 9.5 = -5.52059991329048  -> relu = 0
    #   f4[3] = -10*log10(0.051+0.001) - 9.5 = 3.33996656356849 -> relu = f4[3]
    #   waste = 0 + 3.33996656356849 = 3.33996656356849
    assert waste.item() == pytest.approx(3.33996656356849, abs=1e-3)

    # Pad to J=8 with num_segments=6: boundaries 5 and 6 (0-indexed, both at
    # or past the real 5 boundaries) must contribute EXACTLY 0, since
    # cut_valid masks a_k there to 0 regardless of feature4's value (which
    # is large-but-finite there, since n_next is padded to 0).
    seg_noise8 = torch.cat([seg_noise, torch.zeros(1, 2)], dim=1)
    seg_km8 = torch.cat([seg_km, torch.zeros(1, 2)], dim=1)
    a8, a_phys8, waste8 = head.rollout(seg_noise8, seg_km8, bar, nseg, hard=True)
    assert torch.equal(a8[:, :5], a)
    assert torch.equal(a8[:, 5:], torch.zeros(1, 2))
    assert waste8.item() == pytest.approx(waste.item(), abs=1e-6)

    # num_segments = 1: zero real boundaries, but the loop still computes
    # feature4 at boundary 0 with n_next padded to 0 -- large but finite,
    # and zeroed by cut_valid (a_k == 0 there), not nan/inf.
    head_1seg = AllocationHead()
    a_1seg, _, waste_1seg = head_1seg.rollout(
        torch.tensor([[0.5, 0.0, 0.0]]),
        torch.tensor([[100.0, 0.0, 0.0]]),
        torch.tensor([9.5]),
        torch.tensor([1]),
        hard=True,
    )
    assert torch.equal(a_1seg, torch.zeros(1, 2))
    assert torch.isfinite(waste_1seg)


def test_waste_gradient_pushes_a_wasteful_cut_down():
    """§9 test 7. d(waste)/d(score) is strictly positive at a wasteful
    (feature4 > 0) boundary and EXACTLY zero at a justified (feature4 < 0)
    boundary, because `relu(feature4).detach()` is a frozen constant
    multiplying a_k in the waste sum: a justified boundary's constant is
    exactly 0, so its gradient contribution must be exactly 0 too, not
    merely small.

    JUDGMENT CALL (see task-6-report.md): probes this via `net[-1].bias`
    under the DEFAULT (untrained) init rather than a hand-built weight
    matrix, because default init has `net[-1].weight == 0` exactly, so
    `score == net[-1].bias` for every feature vector with no chain rule
    through the features at all -- `net[-1].bias.grad` after
    `waste.backward()` IS `d(waste)/d(score)`, directly, for a single-
    boundary rollout. Two separate single-demand, single-boundary rollouts
    (fresh head + fresh graph each, so the two contributions are never
    summed into the same backward pass) isolate the wasteful and the
    justified case.
    """
    # Wasteful: tiny noise on both segments -> GSNR after the next segment
    # is still far above a low bar -> feature4 >> 0 -> the cut (which the
    # closed init makes at a ~ sigmoid(-3) ~ 0.047, an interior probability
    # under a soft rollout) was not needed.
    wasteful_head = AllocationHead()
    _, _, waste_wasteful = wasteful_head.rollout(
        torch.tensor([[0.001, 0.001]]), torch.tensor([[100.0, 100.0]]),
        torch.tensor([5.0]), torch.tensor([2]), hard=False,
    )
    waste_wasteful.backward()
    assert wasteful_head.net[-1].bias.grad is not None
    assert wasteful_head.net[-1].bias.grad.item() > 0.0

    # Justified: large noise on both segments against a high bar -> GSNR
    # after the next segment would be below the bar without a cut ->
    # feature4 << 0 -> the cut was genuinely needed.
    justified_head = AllocationHead()
    _, _, waste_justified = justified_head.rollout(
        torch.tensor([[0.5, 0.5]]), torch.tensor([[100.0, 100.0]]),
        torch.tensor([20.0]), torch.tensor([2]), hard=False,
    )
    waste_justified.backward()
    assert justified_head.net[-1].bias.grad is not None
    assert justified_head.net[-1].bias.grad.item() == 0.0


# ---------------------------------------------------------------------------
# alloc_ste: straight-through allocation decisions
#
# The arm exists because the mean-field relaxation and the deployed rollout
# disagree about physics (spec 2.5), and that disagreement manufactures
# VIOLATIONS the deployed network does not have. Measured on constrained_stress
# at the oracle allocation (oracle_gap == 0, hard_num_violated == 0): the soft
# pass reported 10 of 346 demands violated, all 10 feasible in the hard
# rollout, and 100% of the feasibility force at that state came from them.
# Under the STE the forward pass IS the deployed decision, so the two cannot
# disagree; tau then only sets the slope of the backward surrogate.
# ---------------------------------------------------------------------------

def _four_segment_demand():
    seg_db = torch.tensor([[12.0, 9.0, 20.0, 11.0]])
    return (
        db_to_linear_noise(seg_db),
        torch.tensor([[300.0, 400.0, 500.0, 350.0]]),
        torch.tensor([9.5]),
        torch.tensor([4]),
    )


def test_alloc_ste_forward_equals_the_hard_decision():
    """The whole contract: with the arm on, the differentiable pass takes the
    SAME decisions the deployed rollout takes, so the carry it folds is the
    exact chunk noise rather than a partition-weighted expectation."""
    torch.manual_seed(0)
    head = AllocationHead(alloc_ste=True)
    with torch.no_grad():
        head.net[-1].weight.normal_(std=2.0)   # make scores vary in sign
        head.net[-1].bias.zero_()
    seg_noise, seg_km, bar, nseg = _four_segment_demand()

    a_ste, _, _ = head.rollout(seg_noise, seg_km, bar, nseg, tau=1.0)
    a_hard, _, _ = head.rollout(seg_noise, seg_km, bar, nseg, hard=True)

    assert set(a_ste.detach().unique().tolist()) <= {0.0, 1.0}
    assert torch.equal(a_ste.detach(), a_hard)
    # ... and unlike the hard branch it is still attached to the graph.
    assert a_ste.requires_grad
    assert not a_hard.requires_grad


def test_alloc_ste_backward_slope_is_the_sigmoid_surrogate():
    """Forward is a step function, whose true derivative is 0 everywhere. If
    the detach is written wrong the arm silently becomes a hard decision with
    NO gradient, the head stops learning, and every downstream number still
    looks plausible. Pin the surrogate: d a / d score == sigmoid'(s/tau)/tau.

    At init the final layer is zero-weight with bias -3, so score == -3 on
    every boundary regardless of features and d(score)/d(bias) == 1 exactly.
    """
    tau = 0.7
    head = AllocationHead(alloc_ste=True)
    seg_noise = db_to_linear_noise(torch.tensor([[12.0, 9.0, 20.0]]))
    a, _, _ = head.rollout(
        seg_noise, torch.tensor([[300.0, 400.0, 500.0]]),
        torch.tensor([9.5]), torch.tensor([3]), tau=tau,
    )
    assert torch.equal(a.detach(), torch.zeros(1, 2))     # sign of -3

    a.sum().backward()
    sig = 1.0 / (1.0 + math.exp(3.0 / tau))
    expected = 2 * sig * (1.0 - sig) / tau                # two valid boundaries
    assert head.net[-1].bias.grad.item() == pytest.approx(expected, rel=1e-5)
    assert expected > 0.0


def test_alloc_ste_defaults_off_and_is_a_true_no_op():
    """Master's numbers must stay reproducible: the default path is the
    mean-field sigmoid, unchanged."""
    head = AllocationHead()
    assert head.alloc_ste is False
    seg_noise = db_to_linear_noise(torch.tensor([[12.0, 9.0, 20.0]]))
    a, _, _ = head.rollout(
        seg_noise, torch.tensor([[300.0, 400.0, 500.0]]),
        torch.tensor([9.5]), torch.tensor([3]), tau=0.8,
    )
    expected = 1.0 / (1.0 + math.exp(3.0 / 0.8))
    assert torch.allclose(a, torch.full_like(a, expected), atol=1e-6)


def test_alloc_ste_does_not_touch_the_hard_branch():
    """hard_rollout is the DEPLOYED measurement and the selection key. It must
    not move because an optimisation-side flag was flipped, or arms stop being
    comparable to each other."""
    torch.manual_seed(0)
    plain = AllocationHead()
    with torch.no_grad():
        plain.net[-1].weight.normal_(std=2.0)
        plain.net[-1].bias.zero_()
    ste = AllocationHead(alloc_ste=True)
    ste.load_state_dict(plain.state_dict())
    seg_noise, seg_km, bar, nseg = _four_segment_demand()

    a_plain, phys_plain, w_plain = plain.rollout(seg_noise, seg_km, bar, nseg, hard=True)
    a_ste, phys_ste, w_ste = ste.rollout(seg_noise, seg_km, bar, nseg, hard=True)

    assert torch.equal(a_plain, a_ste)
    assert torch.equal(phys_plain, phys_ste)
    assert torch.equal(w_plain, w_ste)


def test_alloc_ste_dropout_keeps_both_decisions_binary():
    """Physics dropout zeroes a_physics, never the priced a. With the STE both
    must stay in {0,1}: a fractional a_physics would put a partial carry reset
    back into the fold, which is the exact defect this arm removes."""
    torch.manual_seed(0)
    head = AllocationHead(alloc_ste=True)
    with torch.no_grad():
        head.net[-1].weight.normal_(std=2.0)
        head.net[-1].bias.fill_(1.0)          # push a healthy share positive
    head.train()
    seg_db = torch.rand(16, 6) * 30.0 - 5.0
    a, a_phys, _ = head.rollout(
        db_to_linear_noise(seg_db),
        torch.rand(16, 6) * 400.0 + 1.0,
        torch.full((16,), 9.5),
        torch.full((16,), 6, dtype=torch.long),
        tau=1.0, dropout_p=0.5,
    )
    assert set(a.detach().unique().tolist()) <= {0.0, 1.0}
    assert set(a_phys.detach().unique().tolist()) <= {0.0, 1.0}
    assert a_phys.sum() < a.sum()             # dropout actually dropped something


def test_alloc_ste_composes_with_greedy_residual():
    """The arm the sweep actually runs is alloc_ste + greedy_residual. Its
    soft pass must equal its own hard rollout; combined with
    test_greedy_residual_init_reproduces_the_oracle (which pins the hard
    rollout to oracle_allocation bit for bit), that makes the DIFFERENTIABLE
    pass reproduce the oracle at init -- the state the convergence
    investigation cares about."""
    torch.manual_seed(0)
    head = AllocationHead(greedy_residual=True, alloc_ste=True)
    seg_db = torch.rand(32, 6) * 30.0 - 5.0
    seg_noise = db_to_linear_noise(seg_db)
    seg_km = torch.rand(32, 6) * 400.0 + 1.0
    bar = torch.rand(32) * 10.0 + 5.0
    nseg = torch.randint(1, 7, (32,))

    a_ste, _, _ = head.rollout(seg_noise, seg_km, bar, nseg, tau=1.0)
    a_hard, _, _ = head.rollout(seg_noise, seg_km, bar, nseg, hard=True)
    assert torch.equal(a_ste.detach(), a_hard)
