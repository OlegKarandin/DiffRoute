"""AllocationHead: per-(demand, boundary) regenerator allocation.

Replaces RegenPlacement, whose single (num_nodes,) logit vector priced
SITES. A site is not what a network buys: 40 demands regenerating at node 7
need 40 devices. The metric change forces the parameterization change, since
a per-node vector cannot express "demand 3 regenerates here, demand 9 does
not".

Three properties are load-bearing and each has a test:

  amortized   the head scores route-local FEATURES, never a demand id, so
              --holdout stays meaningful and Stage IV can score backup
              routes for demands it never trained on (spec section 6 hook 1);

  autoregressive
              a cut's value depends on noise accumulated since the PREVIOUS
              cut, so the walk carries `c` and resets it on a cut. This is
              also what breaks the symmetry between adjacent candidates that
              made the old head place 3/4/5 instead of 4/7/10;

  closed at init
              the final layer is zero-weight, bias -3.0, so a = sigmoid(-3)
              ~ 0.047 deterministically regardless of features. This is what
              lets lambda_dev be live from epoch 0 with no warm-up. Spec 2.2
              traces the warm-up alternative to saturation at ~2000 devices
              against an oracle needing ~60-100, with no gradient left to
              escape. Do not "helpfully" restore a warm-up.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from diffopt.qot.segment_combiner import GSNR_MAX, GSNR_MIN, db_to_linear_noise

# Feature layout, spec 2.1. Order is load-bearing: LOOKAHEAD_COLS indexes it,
# and tests/test_allocation_head.py hand-builds rows against it.
#   0  g_k      = -10 log10(c)                  current chunk GSNR proxy, dB
#   1  bar_d                                    the bar, dB
#   2  h_k      = g_k - bar_d                   headroom now
#   3  -10 log10(n_{k+1})                       next segment, dB       [lookahead]
#   4  -10 log10(c + n_{k+1}) - bar_d           headroom AFTER next    [lookahead]
#   5  km_since_cut / KM_SCALE
#   6  km_remaining / KM_SCALE
#   7  (K_d - k) / K_SCALE
ALLOC_FEATURE_DIM = 8
LOOKAHEAD_COLS = (3, 4)

# Feature 4 is what makes the greedy-optimal policy exactly representable:
# cut at k iff feature 4 < 0. tests/test_oracle.py's representability test
# hand-sets exactly that rule.

KM_SCALE = 1000.0    # a 3000 km path lands feature 6 at 3.0
K_SCALE = 10.0       # K_d runs ~7-19 on ind_132
_EPS = 1e-12         # matches DiffONetPipeline._proxy_eps


class AllocationHead(nn.Module):
    """Score each (demand, boundary) pair, autoregressively along the path."""

    def __init__(
        self,
        hidden: int = 32,
        *,
        lookahead: bool = True,
        init_bias: float = -3.0,
    ) -> None:
        super().__init__()
        self.lookahead = lookahead
        self.net = nn.Sequential(
            nn.Linear(ALLOC_FEATURE_DIM, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        # Zero weights + a negative bias, NOT a small random init: the head
        # must start closed DETERMINISTICALLY, at the same value on every
        # variable, so epoch 0's device count is 0.047 * (number of
        # boundaries) rather than a seed-dependent number.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, init_bias)

        # Diagnostic only: the worst chunk noise the last rollout saw. Not a
        # parameter, not in the graph, not read by the loss. Registered so a
        # test and scripts/diagnose_alloc_gradient.py can check the carry
        # against the combiner's own fold.
        self.register_buffer(
            "last_max_chunk_noise", torch.zeros(0), persistent=False
        )

    def score(self, feats: torch.Tensor) -> torch.Tensor:
        """(..., ALLOC_FEATURE_DIM) -> (...). Positive means "cut here"."""
        if not self.lookahead:
            # Mask rather than shrink the input layer: the arm sweep flips
            # this per run, and a shape change would make the two arms'
            # checkpoints structurally incompatible for no benefit.
            mask = torch.ones(ALLOC_FEATURE_DIM, device=feats.device, dtype=feats.dtype)
            mask[list(LOOKAHEAD_COLS)] = 0.0
            feats = feats * mask
        return self.net(feats).squeeze(-1)

    def rollout(
        self,
        seg_noise: torch.Tensor,
        seg_km: torch.Tensor,
        bar_db: torch.Tensor,
        num_segments: torch.Tensor,
        *,
        tau: float = 1.0,
        hard: bool = False,
        dropout_p: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Walk every demand's boundaries, vectorized across demands.

        Roughly J_max Python iterations per call (about 12 on
        constrained_stress), not one per (demand, boundary) pair.

        Args:
            seg_noise:    (D, J) linear noise per segment. Entries at or past
                          num_segments[d] are ignored — masked to exactly 0.
            seg_km:       (D, J) segment lengths in km, same masking.
            bar_db:       (D,) threshold(bitrate_d) + margin_db.
            num_segments: (D,) long, each in [1, J].
            tau:          sharpness of sigmoid(score / tau). Passed per call,
                          never stored — same rule as pipeline.forward's.
            hard:         deterministic decisions a_k = 1 if score_k > 0.
                          NOT a threshold on the soft pass: under hard
                          decisions the carry c is the EXACT chunk noise, so
                          the rollout is self-consistent physics. Spec 2.5.
            dropout_p:    training-only probability of zeroing a PHYSICS
                          decision. Ignored under .eval() and when hard.

        Returns:
            (a_priced, a_physics), both (D, J-1). They differ only under
            dropout: a_priced is what lambda_dev multiplies, a_physics is
            what the combiner folds and what resets the carry.
        """
        d, j = seg_noise.shape
        device = seg_noise.device
        dtype = seg_noise.dtype

        positions = torch.arange(j, device=device)
        seg_valid = (positions.unsqueeze(0) < num_segments.unsqueeze(1)).to(dtype)
        # A cut at boundary k is real only if segment k+1 exists. Padded
        # columns are forced to EXACTLY 0 (never an epsilon) — the batched
        # fold's exactness argument depends on a padded cut being no cut.
        cut_valid = seg_valid[:, 1:] if j > 1 else seg_valid[:, :0]

        n = seg_noise * seg_valid
        km = seg_km * seg_valid
        km_total = km.sum(dim=1)

        c = torch.zeros(d, dtype=dtype, device=device)
        km_since = torch.zeros(d, dtype=dtype, device=device)
        km_done = torch.zeros(d, dtype=dtype, device=device)
        max_chunk = torch.zeros(d, dtype=dtype, device=device)

        priced_cols = []
        physics_cols = []
        use_dropout = self.training and dropout_p > 0.0 and not hard

        for k in range(j - 1):
            c = c + n[:, k]
            km_since = km_since + km[:, k]
            km_done = km_done + km[:, k]
            max_chunk = torch.maximum(max_chunk, c)

            n_next = n[:, k + 1]
            g_k = -10.0 * torch.log10(c + _EPS)
            g_next = -10.0 * torch.log10(n_next + _EPS)
            g_after = -10.0 * torch.log10(c + n_next + _EPS)

            feats = torch.stack(
                [
                    g_k,
                    bar_db,
                    g_k - bar_db,
                    g_next,
                    g_after - bar_db,
                    km_since / KM_SCALE,
                    (km_total - km_done) / KM_SCALE,
                    (num_segments.to(dtype) - k) / K_SCALE,
                ],
                dim=1,
            )
            s = self.score(feats)

            if hard:
                a_k = (s > 0).to(dtype)
            else:
                a_k = torch.sigmoid(s / tau)
            a_k = a_k * cut_valid[:, k]

            a_phys = a_k
            if use_dropout:
                keep = (torch.rand_like(a_k) >= dropout_p).to(dtype)
                a_phys = a_k * keep

            priced_cols.append(a_k)
            physics_cols.append(a_phys)
            # The PHYSICS decision resets the carry: under dropout the demand
            # really does lose that regenerator, which is the point — it puts
            # the demand back in the hinge's active region, the only region
            # that produces discriminating gradient.
            c = c * (1.0 - a_phys)

        if j >= 1:
            c = c + n[:, j - 1] if j > 1 else n[:, 0]
            max_chunk = torch.maximum(max_chunk, c)
        self.last_max_chunk_noise = max_chunk.detach()

        if priced_cols:
            return torch.stack(priced_cols, dim=1), torch.stack(physics_cols, dim=1)
        empty = torch.zeros(d, 0, dtype=dtype, device=device)
        return empty, empty


def total_device_cost(alloc_by_node: torch.Tensor) -> torch.Tensor:
    """The scalar lambda_dev multiplies: sum_n sum_d a[d, n].

    Written sum-over-nodes-then-sum-over-demands, and isolated in this one
    named function, because Stage IV's shared-restoration objective is
    `sum_n max_s sum_d a[d,s,n]` — algebraically identical to this at
    |S| = 1, not an approximation of it. Keeping the aggregation in one place
    with this shape means adding the scenario axis is a local edit here
    rather than a hunt through the loss. Spec section 6 hook 2.
    """
    return alloc_by_node.sum(dim=0).sum()


def site_view(alloc_by_node: torch.Tensor) -> torch.Tensor:
    """max_d a[d, n] — "is there a regenerator at node n at all".

    DIAGNOSTIC ONLY. Never priced, never in the selection key, never in the
    loss. It exists so the placement trajectory stays readable against the
    pre-Stage-II logs and so a human can see the network shape. Pricing this
    is the exact bug this whole stage removes (spec decision 4).
    """
    return alloc_by_node.max(dim=0).values
