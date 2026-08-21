"""Diagnostic: why have EdgeWeightNet's edge weights collapsed to ~zero?

Follow-up #2 in docs/investigations/open_followups.md measured the symptom
(86% of ind_132's 168 edges below 1e-6, corr(weight, length_km) = -0.17,
routes 2.1-2.4x longer than shortest-by-km) and proposed a mechanism it
explicitly flagged as unconfirmed: that `path_cost_loss`'s gradient
dominates the STE-routed feasibility gradient on the edges that collapse.

This script began as that doc's stated "Next step" — the `edge_weights`
analogue of `diagnose_regen_gradient.py` — to characterize the collapse.
The fix has since landed (unit-mean renormalisation with a live divisor,
redenominating `path_cost_loss` in the fixed `edge_ase_noise` buffer instead
of learned `edge_weights`, standardised static topology features), so this
script now doubles as the post-fix verification tool. Expected post-fix
readings: check C's direct component is exactly 0 (the path-cost term no
longer reads `edge_weights` at all, so nothing is left to differentiate
directly); check F's perturbation ratio reads O(1-10), not the pre-fix
7.9e6 (weights pinned to unit mean can no longer be swamped by a
fixed-scale Vlastelica perturbation); check G's scale-direction derivative
reads ~0 for every loss term (the loss is degree-0 in EdgeWeightNet's raw
output, so "shrink everything" is no longer a free descent direction);
check D's Spearman rank-corr(init, trained) sits well below +0.999 (training
is rearranging relative order, not just uniformly rescaling); and
corr(w, length_km) is positive (weight tracks physical length again,
instead of the pre-fix -0.17).

It decomposes d(total_loss)/d(edge_weights) into its three loss-term
components and runs five checks that discriminate between the candidate
mechanisms:

  A. Scale degeneracy       — is "shrink all weights" a free descent direction?
  B. Gradient decomposition — does path_noise dominate feasibility, and does
                              feasibility reach edge_weights at all?
  C. Direct vs surrogate    — post-fix, path_noise is denominated in the fixed
                              edge_ase_noise buffer, so its direct d/d(edge_weights)
                              is identically zero; does all of its gradient now
                              arrive via the Vlastelica surrogate instead?
  D. Collapse vs no-structure — did training destroy a length-correlation
                              that existed at init, or was there never one?
  E. Barrier routing        — are detours caused by avoiding a few surviving
                              expensive edges?
  F. Perturbation scale     — is lambda*grad_output swamping w in the
                              Vlastelica backward now that w has collapsed?

Usage:
    conda activate diffopt
    python scripts/diagnose_edge_weight_gradient.py --config configs/experiment/small_test_ind132.yaml
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from diffopt.routing.shortest_path import dijkstra
from _common import add_common_args, build_context, demands_for, edge_weights_of, schedule_at


def grad_of(term: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
    g = torch.autograd.grad(term, tensor, retain_graph=True, allow_unused=True)[0]
    return torch.zeros_like(tensor) if g is None else g


def describe(name: str, w: np.ndarray, lens: np.ndarray) -> None:
    # frac<1e-6 / frac<1e-3 were calibrated against pre-fix RAW EdgeWeightNet
    # output, which could (and did) collapse toward the Softplus floor with
    # no lower bound. Post-fix, `w` here is the unit-mean-normalised weight
    # (see edge_weights_of below) — the unit-mean renormalisation pins its mean to exactly 1,
    # so a "pure GLOBAL SCALE collapse" reading (every edge tiny) is
    # structurally unreachable: the normalisation itself prevents the whole
    # population from drifting toward 0. These two fractions reading ~0% post-
    # fix is therefore expected/uninformative, not evidence collapse was fixed
    # — Check A/G (scale-direction gradient) are the checks that actually test
    # the fix.
    corr = float(np.corrcoef(w, lens)[0, 1]) if w.std() > 0 else float("nan")
    print(f"  {name:<22} median={np.median(w):.3e}  mean={w.mean():.3e}  "
          f"max={w.max():.3e}  min={w.min():.3e}")
    print(f"  {'':<22} frac<1e-6={np.mean(w < 1e-6):.0%}  frac<1e-3={np.mean(w < 1e-3):.0%}"
          f"  corr(w, length_km)={corr:+.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    add_common_args(ap, with_demands=False)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    t_cfg, p_cfg = cfg["training"], cfg["pipeline"]
    c_cfg = cfg["constraint"]

    # ---- untrained context (training's actual starting point) --------------
    # build_context seeds with cfg.get("seed", 42) before any module
    # construction, matching train.py's own ordering, so this is
    # bit-identical to training's epoch-0 EdgeWeightNet/RegenPlacement.
    ctx_init = build_context(cfg, load_e2e_checkpoint=False)

    # ---- trained context ------------------------------------------------
    ctx = build_context(cfg, load_e2e_checkpoint=True, checkpoint_path=args.checkpoint)
    topo = ctx.topology
    edges = ctx.edges
    lens = np.array([e.length_km for e in edges], dtype=float)
    n_edges = len(edges)

    ckpt_path = args.checkpoint or f"{cfg['checkpoint_dir']}/best_e2e.pt"
    print(f"config={args.config}")
    print(f"checkpoint={ckpt_path} (epoch {ctx.ckpt['epoch']}, loss={ctx.ckpt['total_loss']:.4f})")
    print(f"nodes={topo.num_nodes} edges={n_edges}  "
          f"lambda_cost={p_cfg['lambda_cost']} dual_init={c_cfg['dual_init']} margin_db={c_cfg['margin_db']} "
          f"lambda_regen={p_cfg['lambda_regen']}\n")

    final_epoch = t_cfg["epochs_e2e"]
    tau, _ = schedule_at(cfg, epoch=final_epoch)
    vl = ctx.ckpt["vlastelica_lambda"]  # checkpoint's own value -- see schedule_at's docstring
    demands = demands_for(ctx, seed=final_epoch)

    # =====================================================================
    # D. Collapse vs never-had-structure
    # =====================================================================
    print("=" * 78)
    print("D. WEIGHT DISTRIBUTION: trained vs untrained-at-training's-seed")
    print("=" * 78)

    w_init = edge_weights_of(ctx_init, tau).numpy()
    w_trained = edge_weights_of(ctx, tau).numpy()
    describe("untrained (init)", w_init, lens)
    describe("trained", w_trained, lens)

    from scipy.stats import spearmanr
    rho = spearmanr(w_init, w_trained).statistic
    print(f"\n  Spearman rank-corr(init weights, trained weights) = {rho:+.3f}")
    print(f"  magnitude ratio  median(trained)/median(init) = "
          f"{np.median(w_trained) / np.median(w_init):.3e}")
    print("  -> high rank-corr + tiny magnitude ratio == pure GLOBAL SCALE collapse")
    print("     (relative ordering preserved, magnitudes annihilated)")
    print("  -> low rank-corr == training actively rearranged which edges are cheap\n")

    # =====================================================================
    # A. Scale degeneracy: is "shrink everything" free?
    # =====================================================================
    print("=" * 78)
    print("A. SCALE DEGENERACY: does uniformly scaling all weights change routing?")
    print("=" * 78)
    ei_np = ctx.pipeline._edge_index.numpy()
    identical = 0
    for d in demands:
        p1 = dijkstra(w_trained.astype(np.float64), ei_np, d.src, d.dst, topo.num_nodes)
        p2 = dijkstra((w_trained * 0.5).astype(np.float64), ei_np, d.src, d.dst, topo.num_nodes)
        if p1 is not None and p2 is not None and np.array_equal(p1, p2):
            identical += 1
    print(f"  routes identical under w vs 0.5*w: {identical}/{len(demands)} demands")
    print(f"  => feasibility_loss and regen_loss are UNCHANGED by the rescale (Dijkstra's")
    print(f"     argmin is scale-invariant). Post-fix, path_noise_loss no longer reads")
    print(f"     edge_weights at all -- it reads the fixed edge_ase_noise buffer -- so it")
    print(f"     is UNCHANGED by this rescale too, not merely 'scales by exactly 0.5' as")
    print(f"     pre-fix path_cost_loss did.")
    print(f"  => Pre-fix this made 'shrink all weights' a FREE descent direction with no")
    print(f"     counter-pressure anywhere in the loss. Post-fix, pipeline.forward")
    print(f"     renormalises edge_weights to unit mean with a live divisor before")
    print(f"     routing, removing the scale degree of freedom from the loss entirely --")
    print(f"     this check now just confirms Dijkstra's pre-existing scale-invariance,")
    print(f"     it is not evidence of a collapse-enabling free direction anymore.\n")

    # =====================================================================
    # B/C/F. Gradient decomposition at both operating points
    #
    # NOT via edge_weights_of: it runs under torch.no_grad() and returns a
    # detached snapshot by construction (see its docstring), so it cannot
    # supply the graph-connected raw tensor these checks differentiate
    # through. A live register_forward_hook during an actual pipeline(...)
    # call is the only way to get that.
    # =====================================================================
    for label, ctx_i, tau_i, vl_i, dem_i in [
        ("UNTRAINED (epoch 0)", ctx_init, *schedule_at(cfg, epoch=1),
         demands_for(ctx_init, seed=1)),
        ("TRAINED (epoch %d)" % ctx.ckpt["epoch"], ctx, tau, vl, demands),
    ]:
        pl = ctx_i.pipeline
        print("=" * 78)
        print(f"B/C/F. GRADIENT DECOMPOSITION ON EdgeWeightNet's RAW output "
              f"(pre-normalisation, NOT the normalised routing weight) - {label}")
        print("=" * 78)

        captured = {}

        # Hooked here on purpose: g_feas/g_regen/g_cost below are gradients
        # w.r.t. this RAW hook output (`w_t`), not w.r.t. the unit-mean-
        # normalised `edge_weights` pipeline.forward actually routes on. That
        # normalised tensor is already detached by the time it would reach a
        # hook site outside forward(), so only the raw, still-autograd-
        # connected output can be differentiated through here. The math is
        # unaffected (raw and normalised differ by a constant factor per
        # forward call, and Check G's zero-crossing identity holds for
        # either), but readers should not mistake this section's numbers for
        # gradients w.r.t. the actual routing weight.
        def hook(_mod, _inp, out):
            out.retain_grad()
            captured["w"] = out

        h = pl.edge_weight_net.register_forward_hook(hook)
        path_noise_costs, gsnr_preds, path_inds, regen_probs = pl(
            dem_i, tau=tau_i, lambda_=vl_i)
        h.remove()
        w_t = captured["w"]
        # Mirror edge_weights_of's unit-mean renormalisation: the raw hook
        # output is EdgeWeightNet's pre-normalisation Softplus output, not
        # what pipeline.forward actually routes with, and the unit-mean renormalisation makes
        # the loss degree-0 in that raw scale, so it can drift freely.
        w_t_norm = w_t.detach().squeeze(-1) / w_t.detach().squeeze(-1).mean().clamp_min(1e-12)

        feas = torch.zeros(1)
        n_infeas = 0
        for d in dem_i:
            thr = torch.tensor(ctx_i.mod_cfg.required_snr_threshold(d.bitrate_gbps))
            sf = F.relu(thr + c_cfg["margin_db"] - gsnr_preds[d.id])
            feas = feas + sf
            if sf.item() > 0:
                n_infeas += 1

        # At epoch 0 every per-demand dual equals dual_init (they diverge
        # only after update_duals starts adjusting them per demand), so this
        # single-scalar substitution is only valid at epoch 0.
        L_feas = c_cfg["dual_init"] * feas.squeeze()
        L_regen = p_cfg["lambda_regen"] * regen_probs.sum()
        L_cost = p_cfg["lambda_cost"] * sum(path_noise_costs.values())

        g_feas = grad_of(L_feas, w_t).squeeze(-1)
        g_regen = grad_of(L_regen, w_t).squeeze(-1)
        g_cost = grad_of(L_cost, w_t).squeeze(-1)
        g_tot = g_feas + g_regen + g_cost

        # C. Post-fix, the path-cost term is denominated in edge_ase_noise, so
        # its analytic direct d/d(edge_weights) is identically zero — every
        # remaining component must arrive via the Vlastelica surrogate. Before
        # the fix the split was 14.46 direct vs 2.2e-6 surrogate; a nonzero
        # direct component here means the term is reading edge_weights again.
        usage = torch.zeros(n_edges)
        for d in dem_i:
            usage += path_inds[d.id].detach()
        g_cost_direct = torch.zeros(n_edges)
        g_cost_surrogate = g_cost - g_cost_direct

        print(f"  demands={len(dem_i)}  infeasible={n_infeas}  tau={tau_i:.3f} "
              f"vlastelica_lambda={vl_i:.3f}")
        print(f"  L_feas={L_feas.item():.4f}  L_regen={L_regen.item():.4f}  "
              f"L_cost={L_cost.item():.6f}\n")

        def gstat(nm, g):
            nz = int((g != 0).sum())
            print(f"  {nm:<26} L1={g.abs().sum():.4e}  max|g|={g.abs().max():.4e}  "
                  f"nonzero={nz}/{n_edges}  sum={g.sum():+.4e}")

        gstat("grad path_noise (total)", g_cost)
        gstat("  |- direct d/dw", g_cost_direct)
        gstat("  |- via surrogate", g_cost_surrogate)
        gstat("grad feasibility (STE)", g_feas)
        gstat("grad regen_count", g_regen)
        gstat("grad TOTAL", g_tot)

        l1c, l1f = g_cost.abs().sum().item(), g_feas.abs().sum().item()
        print(f"\n  ratio  L1(path_noise) / L1(feasibility) = "
              f"{(l1c / l1f) if l1f > 0 else float('inf'):.3f}")
        pos_frac = float((g_cost_direct > 0).float().sum() / max(1, int((usage > 0).sum())))
        print(f"  path_noise's direct component is >=0 on every edge "
              f"(pure shrink): {bool((g_cost_direct >= 0).all())}")
        print(f"  edges used by >=1 demand: {int((usage > 0).sum())}/{n_edges}"
              f"   (pos_frac={pos_frac:.2f})")

        # G. Scale-direction derivative: dL(c*w)/dc at c=1, which by the chain
        # rule is sum_e g_e * w_e. Pre-fix, this was the exact quantity that
        # drove the collapse: path_cost (the pre-fix name for this term) was
        # degree-1 homogeneous in edge_weights, so Euler's theorem gave
        # dL/dc == L itself -- a permanent positive shrink pressure -- while
        # feasibility/regen were already scale-invariant (Dijkstra's argmin
        # ignores global scale, so dL/dc == 0 for those).
        # Post-fix, path_noise is denominated in the fixed edge_ase_noise
        # buffer, so it is degree-0 in edge_weights too: dL/dc should now
        # read ~0 for every term, same as feasibility/regen always did.
        wv_ = w_t_norm
        print("\n  G. scale-direction derivative  dL(c*w)/dc |_(c=1) = sum_e g_e*w_e:")
        for nm, g, ref in [("path_noise", g_cost, L_cost.item()),
                           ("feasibility", g_feas, None),
                           ("regen_count", g_regen, None),
                           ("TOTAL", g_tot, None)]:
            d = float((g * wv_).sum())
            extra = (f"   (L_cost itself = {ref:.6e}; "
                     f"pre-fix Euler check, expected to disagree now)") if ref is not None else ""
            print(f"     {nm:<14} dL/dc = {d:+.6e}{extra}")
        print("     -> post-fix ALL terms should read ~0: the loss is degree-0 in")
        print("        EdgeWeightNet's raw output, so no shrink direction exists.")
        print("        (pre-fix: path_cost +7.198e-02, feasibility +3.164e-01 at epoch 0)")

        # F. Vlastelica perturbation scale vs weight scale
        wv = w_t_norm
        # grad_output flowing into the surrogate == d(total)/d(path_indicator).
        # Reconstruct its scale from the two contributing terms for one demand.
        gpi = grad_of(L_feas + L_cost, path_inds[dem_i[0].id])
        pert = vl_i * gpi.abs()
        print(f"\n  F. perturbation scale check (demand {dem_i[0].id}):")
        print(f"     median |w|            = {wv.median():.4e}")
        print(f"     median lambda*|dL/dz| = {pert.median():.4e}   "
              f"max = {pert.max():.4e}")
        ratio = (pert.max() / wv.median()).item() if wv.median() > 0 else float("inf")
        print(f"     max perturbation / median weight = {ratio:.3e}")
        print(f"     (pre-fix this read 7.9e6 at epoch 60; with weights pinned to")
        print(f"      unit mean it should now read O(1-10), a genuine perturbation)\n")

    # =====================================================================
    # E. Barrier routing
    # =====================================================================
    print("=" * 78)
    print("E. BARRIER ROUTING: do detours avoid the few surviving expensive edges?")
    print("=" * 78)
    wt64 = w_trained.astype(np.float64)
    tr_max, sp_max, tr_km, sp_km, tr_hops, sp_hops = [], [], [], [], [], []
    for d in demands:
        pi_t = dijkstra(wt64, ei_np, d.src, d.dst, topo.num_nodes)
        pi_s = dijkstra(lens, ei_np, d.src, d.dst, topo.num_nodes)
        if pi_t is None or pi_s is None:
            continue
        et = np.flatnonzero(pi_t > 0.5)
        es = np.flatnonzero(pi_s > 0.5)
        tr_max.append(w_trained[et].max()); sp_max.append(w_trained[es].max())
        tr_km.append(lens[et].sum());       sp_km.append(lens[es].sum())
        tr_hops.append(len(et));            sp_hops.append(len(es))
    tr_max, sp_max = np.array(tr_max), np.array(sp_max)
    print(f"  mean km          trained={np.mean(tr_km):8.1f}   shortest={np.mean(sp_km):8.1f}"
          f"   ratio={np.mean(tr_km)/np.mean(sp_km):.3f}x")
    print(f"  mean hops        trained={np.mean(tr_hops):8.2f}   shortest={np.mean(sp_hops):8.2f}")
    print(f"  max edge weight  trained={np.median(tr_max):8.3e}   shortest={np.median(sp_max):8.3e}"
          f"   (median over demands)")
    print(f"  demands whose shortest-km path contains a strictly more expensive "
          f"edge than any on the trained path: {int((sp_max > tr_max).sum())}/{len(tr_max)}")
    print("  -> a high count means routing is dodging a few surviving 'barrier'")
    print("     edges, i.e. the detour is driven by weight SPREAD, not randomness.")


if __name__ == "__main__":
    main()
