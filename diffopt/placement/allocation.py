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

              greedy_residual=True carve-out (spec 5.2): the closed point
              moves, but the CLOSED-ness doesn't. With greedy_residual, the
              final layer is zero-weight AND zero-bias (MLP == 0 exactly),
              and the score is instead `alpha * (bar_d - g_after)` — a
              linear combination of features 1 and 4 that the OLD zero-weight
              parameterization could already express, just not at a useful
              point. So this changes WHERE the closed init sits (exactly the
              oracle's own greedy cut, `score > 0 <=> feature4 < 0`, boundary
              for boundary) and not WHAT the head can represent, nor whether
              lambda_dev is live from epoch 0: it still is, with no warm-up
              schedule, from the very first step. This is a different regime
              from the failure mode above (which opens near sigmoid(0) ~ 0.5
              everywhere, i.e. ~2000 saturated devices with no gradient to
              escape) — the two are observably distinguishable via
              alloc_score_min/max and the alloc_alpha CSV column, and neither
              is a warm-up: nothing here anneals or ramps toward this point,
              it IS the point from epoch 0.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

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
ROUTE_CONTEXT_COLS = (5, 6, 7)

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
        route_context: bool = True,
        greedy_residual: bool = False,
        init_bias: float = -3.0,
    ) -> None:
        super().__init__()
        self.lookahead = lookahead
        self.route_context = route_context
        self.greedy_residual = greedy_residual
        if greedy_residual and not lookahead:
            raise ValueError(
                "greedy_residual needs feature 4, which lookahead=False masks"
            )
        self.net = nn.Sequential(
            nn.Linear(ALLOC_FEATURE_DIM, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        if greedy_residual:
            # softplus keeps alpha > 0 so the sign test never inverts.
            # softplus(2.949) = 3.0
            self.alpha_raw = nn.Parameter(torch.tensor(2.949))
            # Zero weight AND zero bias, NOT init_bias: the linear residual
            # below needs MLP == 0 exactly, or the closed point is not the
            # oracle's own boundary any more.
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)
        else:
            # Zero weights + a negative bias, NOT a small random init: the
            # head must start closed DETERMINISTICALLY, at the same value on
            # every variable, so epoch 0's device count is 0.047 * (number
            # of boundaries) rather than a seed-dependent number.
            nn.init.zeros_(self.net[-1].weight)
            nn.init.constant_(self.net[-1].bias, init_bias)

        # Diagnostic only: the worst chunk noise the last rollout saw. Not a
        # parameter, not in the graph, not read by the loss. Registered so a
        # test and scripts/diagnose_alloc_gradient.py can check the carry
        # against the combiner's own fold.
        self.register_buffer(
            "last_max_chunk_noise", torch.zeros(0), persistent=False
        )

    @property
    def alpha(self) -> float:
        """softplus(alpha_raw) when greedy_residual, else nan.

        Always safely callable regardless of the flag — train.py's CSV-row
        code reads this every epoch unconditionally.
        """
        return float(F.softplus(self.alpha_raw).detach()) if self.greedy_residual else float("nan")

    def score(self, feats: torch.Tensor) -> torch.Tensor:
        """(..., ALLOC_FEATURE_DIM) -> (...). Positive means "cut here"."""
        # The residual reads feature 4 from the UNMASKED feats, before the
        # lookahead/route_context mask below (which only ever applies to the
        # MLP's input) can touch it. Get this order wrong and
        # greedy_residual=True, route_context=False would silently zero out
        # the residual's own feature 4, even though lookahead=True is
        # required precisely to keep it available.
        residual = 0.0
        if self.greedy_residual:
            residual = -F.softplus(self.alpha_raw) * feats[..., 4]

        if not self.lookahead or not self.route_context:
            # Mask rather than shrink the input layer: the arm sweep flips
            # this per run, and a shape change would make the two arms'
            # checkpoints structurally incompatible for no benefit.
            mask = torch.ones(ALLOC_FEATURE_DIM, device=feats.device, dtype=feats.dtype)
            if not self.lookahead:
                mask[list(LOOKAHEAD_COLS)] = 0.0
            if not self.route_context:
                mask[list(ROUTE_CONTEXT_COLS)] = 0.0
            feats = feats * mask
        return residual + self.net(feats).squeeze(-1)

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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
            (a_priced, a_physics, waste). a_priced/a_physics are both
            (D, J-1) and differ only under dropout: a_priced is what
            lambda_dev multiplies, a_physics is what the combiner folds and
            what resets the carry. waste is a scalar,
            sum_{d,k} a_priced[d,k] * relu(feature4[d,k]), masked by
            cut_valid. Only the relu(feature4) COEFFICIENT is
            gradient-detached, per this module's carry-as-observation
            invariant (deviation 1 in the task-6 brief: an undetached
            relu(feature4) would still carry gradient into n_next even
            though the carry cd is already detached, rewarding routing onto
            noisier next segments) — the priced allocation a_k that
            multiplies it is NOT detached, and legitimately carries gradient
            into n_next/routing via its own dependence on score.
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
        waste = torch.zeros((), dtype=dtype, device=device)

        priced_cols = []
        physics_cols = []
        use_dropout = self.training and dropout_p > 0.0 and not hard

        for k in range(j - 1):
            c = c + n[:, k]
            km_since = km_since + km[:, k]
            km_done = km_done + km[:, k]
            max_chunk = torch.maximum(max_chunk, c)

            n_next = n[:, k + 1]
            # The carry enters the features as an OBSERVATION, never as a
            # differentiable function of this head's own earlier decisions.
            # Undetached, d a_{k+1} / d a_k runs through -10 log10(c) and is
            # both huge (c_k / c_{k+1} ~ 1e2-1e3 right after a cut) and
            # sign-indefinite, which breaks "regen helps" (invariants.md) at
            # the level of the head's parameters: measured on
            # constrained_stress's epoch-35 checkpoint, d(device_count)/d(bias)
            # swings -3317 .. +1163 across a 0.002-wide window in that one
            # parameter -- i.e. lambda_dev rewarding MORE devices -- where the
            # detached value is a steady +89 .. +92. Forward values are
            # unchanged; c = c * (1 - a_phys) below stays live physics.
            cd = c.detach()
            g_k = -10.0 * torch.log10(cd + _EPS)
            g_next = -10.0 * torch.log10(n_next + _EPS)
            g_after = -10.0 * torch.log10(cd + n_next + _EPS)

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

            waste = waste + (a_k * F.relu(g_after - bar_db).detach()).sum()

            a_phys = a_k
            if use_dropout:
                keep = (torch.rand_like(a_k) >= dropout_p).to(dtype)
                a_phys = a_k * keep

            priced_cols.append(a_k)
            physics_cols.append(a_phys)
            # The PHYSICS decision resets the carry: under dropout the demand
            # really does lose that regenerator, which is the point — it puts
            # the demand back in the hinge's active region, the only region
            # that produces discriminating gradient. Feature 5 (km_since_cut)
            # resets the same way: it is the distance since the PREVIOUS cut,
            # not since the start of the route, or it duplicates feature 6.
            c = c * (1.0 - a_phys)
            km_since = km_since * (1.0 - a_phys)

        if j >= 1:
            c = c + n[:, j - 1] if j > 1 else n[:, 0]
            max_chunk = torch.maximum(max_chunk, c)
        self.last_max_chunk_noise = max_chunk.detach()

        if priced_cols:
            return torch.stack(priced_cols, dim=1), torch.stack(physics_cols, dim=1), waste
        empty = torch.zeros(d, 0, dtype=dtype, device=device)
        return empty, empty, torch.zeros((), dtype=dtype, device=device)


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
