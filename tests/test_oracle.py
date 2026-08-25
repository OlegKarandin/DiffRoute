"""The oracle: exact minimum devices on a FIXED route.

Spec 2.6. oracle_gap == 0 is the load-bearing acceptance test for the whole
stage, so these tests are about the oracle actually being the minimum —
not merely about it returning something plausible.
"""
import itertools

import pytest
import torch

from diffopt.placement.allocation import AllocationHead, ALLOC_FEATURE_DIM
from diffopt.placement.oracle import (
    oracle_allocation,
    oracle_gap,
    repair_with_oracle,
)
from diffopt.qot.segment_combiner import SegmentCombiner


def brute_force_min(seg_db, bar):
    """Smallest number of cuts making every chunk clear `bar`, by enumeration.
    Exponential — only ever called on <= 12 segments in these tests."""
    n = [10.0 ** (-g / 10.0) for g in seg_db]
    bar_noise = 10.0 ** (-bar / 10.0)
    best = None
    for r in range(len(seg_db)):
        for cuts in itertools.combinations(range(len(seg_db) - 1), r):
            chunks, start = [], 0
            for c in sorted(cuts) + [len(seg_db) - 1]:
                chunks.append(sum(n[start : c + 1]))
                start = c + 1
            if all(ch <= bar_noise * (1 + 1e-12) for ch in chunks):
                best = r
                break
        if best is not None:
            break
    return best


def test_no_cuts_needed_when_the_whole_path_clears():
    res = oracle_allocation(
        torch.tensor([[20.0, 20.0, 20.0]]), torch.tensor([12.0]), torch.tensor([3])
    )
    assert torch.equal(res.a, torch.zeros(1, 2))
    assert res.count.tolist() == [0]
    assert res.feasible.tolist() == [True]


def test_cuts_exactly_where_the_bar_would_bust():
    """Three equal 12 dB segments, bar 9.5 dB. Two of them sum to 9.0 dB
    (below the bar), so a cut is needed before the third but not the second."""
    res = oracle_allocation(
        torch.tensor([[12.0, 12.0, 12.0]]), torch.tensor([8.5]), torch.tensor([3])
    )
    assert res.a.tolist() == [[0.0, 1.0]]
    assert res.count.tolist() == [1]


def test_a_segment_that_alone_busts_the_bar_is_infeasible():
    """No allocation can rescue it — cutting on both sides still leaves that
    segment as its own chunk. The oracle must SAY so rather than return a
    plausible count."""
    res = oracle_allocation(
        torch.tensor([[20.0, 5.0, 20.0]]), torch.tensor([9.5]), torch.tensor([3])
    )
    assert res.feasible.tolist() == [False]


@pytest.mark.parametrize("seed", range(20))
def test_oracle_matches_brute_force_on_random_paths(seed):
    """The exchange argument, checked rather than assumed."""
    g = torch.Generator().manual_seed(seed)
    n = int(torch.randint(2, 9, (1,), generator=g).item())
    seg_db = (torch.rand(n, generator=g) * 8.0 + 13.0).tolist()
    bar = 12.0
    expected = brute_force_min(seg_db, bar)
    if expected is None:
        pytest.skip("randomly drew an infeasible path")
    res = oracle_allocation(
        torch.tensor([seg_db]), torch.tensor([bar]), torch.tensor([n])
    )
    assert res.count.item() == expected


def test_ragged_batch_matches_per_demand_calls():
    seg = torch.tensor([
        [14.0, 14.0, 14.0, 14.0],
        [20.0, 20.0,  0.0,  0.0],
        [13.0, 13.0, 13.0,  0.0],
    ])
    bar = torch.tensor([11.0, 11.0, 11.0])
    nseg = torch.tensor([4, 2, 3])
    batched = oracle_allocation(seg, bar, nseg)
    for row in range(3):
        k = int(nseg[row])
        one = oracle_allocation(
            seg[row : row + 1, :k], bar[row : row + 1], nseg[row : row + 1]
        )
        assert one.count.item() == batched.count[row].item()
        assert torch.equal(one.a[0], batched.a[row, : max(k - 1, 0)])


def test_padded_columns_never_carry_a_cut():
    res = oracle_allocation(
        torch.tensor([[14.0, 14.0, 14.0, 14.0]]),
        torch.tensor([11.0]),
        torch.tensor([2]),
    )
    assert torch.equal(res.a[:, 1:], torch.zeros(1, 2))


def test_greedy_weights_reproduce_the_oracle():
    """Spec section 8 item 2, the architecture-vs-optimisation separator.

    Feature 4 is `-10 log10(c + n_next) - bar`, i.e. the headroom AFTER
    absorbing the next segment. The greedy rule is exactly "cut iff feature
    4 < 0", so a head whose final layer is -M * e_4 reproduces the oracle
    bit for bit. If this passes and a real run still shows oracle_gap > 0,
    the gap is an OPTIMISATION failure and no amount of architecture change
    will close it.
    """
    head = AllocationHead()
    with torch.no_grad():
        # Create weight matrix with correct shape for final layer (1, 32)
        w = torch.zeros(1, head.net[-1].weight.shape[1])
        w[0, 4] = -1.0e3
        head.net[-1].weight.copy_(w)
        head.net[-1].bias.zero_()
        # Make the first two layers the identity on the feature vector so the
        # hand-set final layer sees the raw features.
        head.net[0].weight.zero_()
        head.net[0].weight[:ALLOC_FEATURE_DIM, :].copy_(torch.eye(ALLOC_FEATURE_DIM))
        head.net[0].bias.fill_(50.0)       # keep ReLU in its linear region
        head.net[2].weight.zero_()
        head.net[2].weight.copy_(torch.eye(head.net[2].weight.shape[0]))
        head.net[2].bias.zero_()
        # Undo the +50 offset that carried through both identity layers.
        head.net[-1].bias.copy_((-w @ torch.full((w.shape[1], 1), 50.0)).squeeze())

    torch.manual_seed(0)
    seg_db = torch.rand(16, 6) * 8.0 + 13.0
    bar = torch.full((16,), 12.0)
    nseg = torch.full((16,), 6, dtype=torch.long)
    seg_noise = 10.0 ** (-seg_db / 10.0)

    a_head, _, _ = head.rollout(seg_noise, torch.zeros(16, 6), bar, nseg, hard=True)
    expected = oracle_allocation(seg_db, bar, nseg)
    assert torch.equal(a_head, expected.a)


def test_oracle_gap_is_zero_when_the_head_matches_and_positive_otherwise():
    a_oracle = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    count = torch.tensor([1, 1])
    assert oracle_gap(a_oracle, count) == 0
    assert oracle_gap(torch.tensor([[1.0, 1.0], [1.0, 0.0]]), count) == 1


def test_repair_makes_every_repairable_demand_feasible():
    """Spec 2.6's deployment guarantee: after repair, a demand is violated
    only if its ROUTE admits no feasible allocation at all."""
    seg = torch.tensor([[12.0, 12.0, 12.0], [20.0, 20.0, 20.0]])
    bar = torch.tensor([9.5, 9.5])
    nseg = torch.tensor([3, 3])
    hard_a = torch.zeros(2, 2)              # head placed nothing: row 0 fails
    a_rep, gsnr_rep, feasible = repair_with_oracle(
        seg, bar, nseg, hard_a, SegmentCombiner()
    )
    assert feasible.tolist() == [True, True]
    assert (gsnr_rep >= bar - 1e-4).all()
    assert torch.equal(a_rep[1], hard_a[1])  # row 1 was already fine: untouched


def test_repair_leaves_an_unrepairable_demand_flagged_not_silently_fixed():
    seg = torch.tensor([[20.0, 5.0, 20.0]])
    res = repair_with_oracle(
        seg, torch.tensor([9.5]), torch.tensor([3]), torch.zeros(1, 2),
        SegmentCombiner(),
    )
    assert res[2].tolist() == [False]
