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

              A `greedy_residual` carve-out (spec 5.2) used to make the
              closed point coincide with the oracle's own greedy cut at
              init. Removed 2026-09 (open_followups.md item #8): it hard-
              codes the `|S| = 1` optimal rule into the score, so a result
              obtained with it on would be "the relaxation recovering the
              optimum because we told it the answer." Recoverable from git
              history if Stage IV restoration needs it back.
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
        alloc_ste: bool = False,
        init_bias: float = -3.0,
    ) -> None:
        super().__init__()
        self.lookahead = lookahead
        self.route_context = route_context
        self.alloc_ste = alloc_ste
        self.net = nn.Sequential(
            nn.Linear(ALLOC_FEATURE_DIM, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
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
        # Diagnostic only, same contract as the buffer above: (D, J-1) raw
        # pre-activation scores from the last rollout, detached.
        #
        # The score cannot be recovered from the allocations the rollout
        # returns. Under `alloc_ste` the forward value is EXACTLY 0 or 1, so
        # inverting the sigmoid pins every boundary to float32's clamps
        # (-87.336545 / +15.942385 in score units at tau=1) and reports
        # nothing about magnitude — which is precisely what a saturation
        # diagnostic needs. Measured on the augmented-Lagrangian gate runs:
        # all three al_ste_greedy seeds logged those two constants for 60/60
        # epochs while the true scores ran to -47.
        # See docs/investigations/augmented_lagrangian_gate.md.
        #
        # Rebound rather than mutated in place (like last_max_chunk_noise),
        # so a caller that kept a reference to an earlier rollout's scores
        # still holds that rollout's values after a later one — the hard
        # rollout in train.py runs after the soft pass whose scores are
        # being logged.
        self.register_buffer("last_scores", torch.zeros(0), persistent=False)

    def score(self, feats: torch.Tensor) -> torch.Tensor:
        """(..., ALLOC_FEATURE_DIM) -> (...). Positive means "cut here"."""
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
                          Under `alloc_ste` it no longer affects the forward
                          value at all, only the slope of the backward
                          surrogate.
            hard:         deterministic decisions a_k = 1 if score_k > 0.
                          NOT a threshold on the soft pass: under hard
                          decisions the carry c is the EXACT chunk noise, so
                          the rollout is self-consistent physics. Spec 2.5.

        Returns:
            (a_priced, a_physics, waste). a_priced/a_physics are both
            (D, J-1): a_priced is what lambda_dev multiplies, a_physics is
            what the combiner folds and what resets the carry. waste is a scalar,
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
        score_cols = []

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
            elif self.alloc_ste:
                # Straight-through. The FORWARD value is the deployed decision,
                # so this pass folds the exact chunk noise instead of a
                # partition-weighted expectation; the BACKWARD pass keeps
                # sigmoid'(s/tau)/tau, so the head still learns.
                #
                # Without it the relaxation and hard_rollout disagree about
                # which demands are feasible, and the duals price violations
                # the deployed network does not have. Measured on
                # constrained_stress at an allocation with oracle_gap == 0 and
                # hard_num_violated == 0: 10 of 346 demands read as violated in
                # the soft pass, every one feasible in the hard rollout, and
                # those 10 carried 100% of the feasibility force -- 9.7x the
                # combined lambda_dev + lambda_waste shed. Annealing tau made
                # it worse (13 phantoms at tau=0.3), because a cut that is
                # barely needed has score ~ 0 and sigmoid(0/tau) == 0.5 at
                # EVERY temperature.
                #
                # tau keeps only its backward role here, so the anneal has no
                # forward job left; the arm pins alloc_tau_end to
                # alloc_tau_start and train.py warns when it is not pinned.
                a_soft = torch.sigmoid(s / tau)
                a_k = a_soft + ((s > 0).to(dtype) - a_soft).detach()
            else:
                a_k = torch.sigmoid(s / tau)
            a_k = a_k * cut_valid[:, k]

            waste = waste + (a_k * F.relu(g_after - bar_db).detach()).sum()

            priced_cols.append(a_k)
            physics_cols.append(a_k)
            score_cols.append(s.detach())
            # The PHYSICS decision resets the carry. Feature 5 (km_since_cut)
            # resets the same way: it is the distance since the PREVIOUS cut,
            # not since the start of the route, or it duplicates feature 6.
            c = c * (1.0 - a_k)
            km_since = km_since * (1.0 - a_k)

        if j >= 1:
            c = c + n[:, j - 1] if j > 1 else n[:, 0]
            max_chunk = torch.maximum(max_chunk, c)
        self.last_max_chunk_noise = max_chunk.detach()

        if priced_cols:
            self.last_scores = torch.stack(score_cols, dim=1)
            return torch.stack(priced_cols, dim=1), torch.stack(physics_cols, dim=1), waste
        empty = torch.zeros(d, 0, dtype=dtype, device=device)
        self.last_scores = empty
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
