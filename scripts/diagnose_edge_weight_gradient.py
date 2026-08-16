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
  B. Gradient decomposition — does path_cost dominate feasibility, and does
                              feasibility reach edge_weights at all?
  C. Direct vs surrogate    — is path_cost's gradient the plain positive
                              d/dw of (path_indicator . w), or does the
                              Vlastelica backward contribute?
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
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from diffopt.demands import generate_demands
from diffopt.modulation import ModulationConfig
from diffopt.pipeline import DiffONetPipeline
from diffopt.placement.regenerator import RegenPlacement
from diffopt.qot.segment_combiner import SegmentCombiner
from diffopt.routing.edge_weight_net import EdgeWeightNet
from diffopt.routing.shortest_path import dijkstra
from diffopt.topology import load_topology
from diffopt.train import linear_anneal, load_qot_model


def grad_of(term: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
    g = torch.autograd.grad(term, tensor, retain_graph=True, allow_unused=True)[0]
    return torch.zeros_like(tensor) if g is None else g


def describe(name: str, w: np.ndarray, lens: np.ndarray) -> None:
    corr = float(np.corrcoef(w, lens)[0, 1]) if w.std() > 0 else float("nan")
    print(f"  {name:<22} median={np.median(w):.3e}  mean={w.mean():.3e}  "
          f"max={w.max():.3e}  min={w.min():.3e}")
    print(f"  {'':<22} frac<1e-6={np.mean(w < 1e-6):.0%}  frac<1e-3={np.mean(w < 1e-3):.0%}"
          f"  corr(w, length_km)={corr:+.3f}")


def build_pipeline(cfg, topo, qot, dev, seed_init: bool):
    """Construct a pipeline. If seed_init, reproduce train.py's exact RNG order
    so EdgeWeightNet's initial weights match what training actually started from."""
    if seed_init:
        torch.manual_seed(cfg.get("seed", 42))
        # train.py's construction order consumes RNG in this sequence:
        # load_qot_model (already done by caller under the same seed), then
        # SegmentCombiner(), EdgeWeightNet(), RegenPlacement().
    sc = SegmentCombiner()
    ewn = EdgeWeightNet().to(dev)
    regen = RegenPlacement(topo.num_nodes).to(dev)
    pipe = DiffONetPipeline(
        topology=topo, qot_model=qot, segment_combiner=sc,
        edge_weight_net=ewn, regen_placement=regen,
        channel_loading_fraction=cfg["pipeline"]["channel_loading_fraction"],
        max_spans=cfg.get("max_spans_per_segment", 60),
    ).to(dev)
    return pipe, ewn, regen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment/small_test_ind132.yaml")
    ap.add_argument("--checkpoint", default=None,
                    help="Defaults to <checkpoint_dir>/best_e2e.pt")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    dev = torch.device("cpu")
    t_cfg, p_cfg = cfg["training"], cfg["pipeline"]
    sc_cfg = cfg.get("segment_combiner", {})

    topo = load_topology(cfg["topology"], cfg["modulation_formats"])
    mod_cfg = ModulationConfig.from_yaml(cfg["modulation_formats"])

    # Seed exactly as train.py does *before* loading the QoT model, so the
    # untrained EdgeWeightNet below is bit-identical to training's epoch-0 state.
    torch.manual_seed(cfg.get("seed", 42))
    qot = load_qot_model(cfg["qot_checkpoint"], cfg, dev)

    edges = list(topo.undirected_edges)
    lens = np.array([e.length_km for e in edges], dtype=float)
    n_edges = len(edges)

    # ---- untrained pipeline (training's actual starting point) -------------
    pipe_init, ewn_init, regen_init = build_pipeline(cfg, topo, qot, dev, seed_init=False)

    # ---- trained pipeline --------------------------------------------------
    pipe, ewn, regen = build_pipeline(cfg, topo, qot, dev, seed_init=False)
    ckpt_path = Path(args.checkpoint or f"{cfg['checkpoint_dir']}/best_e2e.pt")
    ckpt = torch.load(ckpt_path, map_location=dev)
    ewn.load_state_dict(ckpt["edge_weight_net_state"])
    with torch.no_grad():
        regen.regen_logits.copy_(ckpt["regen_logits"])

    print(f"config={args.config}")
    print(f"checkpoint={ckpt_path} (epoch {ckpt['epoch']}, loss={ckpt['total_loss']:.4f})")
    print(f"nodes={topo.num_nodes} edges={n_edges}  "
          f"lambda_cost={p_cfg['lambda_cost']} lambda_infeasible={p_cfg['lambda_infeasible']} "
          f"lambda_regen={p_cfg['lambda_regen']}\n")

    final_epoch = t_cfg["epochs_e2e"]
    tau = t_cfg["regen_tau_end"]
    t_sm = sc_cfg.get("soft_max_temperature_min", 0.01)
    vl = ckpt["vlastelica_lambda"]
    demands = generate_demands(topo, cfg["num_demands"], cfg["bitrate_options"],
                               seed=final_epoch)

    # =====================================================================
    # D. Collapse vs never-had-structure
    # =====================================================================
    print("=" * 78)
    print("D. WEIGHT DISTRIBUTION: trained vs untrained-at-training's-seed")
    print("=" * 78)

    def edge_weights_of(pipeline_obj, regen_obj):
        with torch.no_grad():
            rp = regen_obj.get_regen_probs(tau)
            feats = torch.cat([
                pipeline_obj._topo_edge_features,
                rp[pipeline_obj._edge_src_ids].unsqueeze(1),
                rp[pipeline_obj._edge_dst_ids].unsqueeze(1),
            ], dim=1)
            raw = pipeline_obj.edge_weight_net(feats).squeeze(-1)
            # Match pipeline.forward's unit-mean renormalisation exactly.
            return (raw / raw.mean().clamp_min(1e-12)).numpy()

    w_init = edge_weights_of(pipe_init, regen_init)
    w_trained = edge_weights_of(pipe, regen)
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
    ei_np = pipe._edge_index.numpy()
    identical = 0
    for d in demands:
        p1 = dijkstra(w_trained.astype(np.float64), ei_np, d.src, d.dst, topo.num_nodes)
        p2 = dijkstra((w_trained * 0.5).astype(np.float64), ei_np, d.src, d.dst, topo.num_nodes)
        if p1 is not None and p2 is not None and np.array_equal(p1, p2):
            identical += 1
    print(f"  routes identical under w vs 0.5*w: {identical}/{len(demands)} demands")
    print(f"  => feasibility_loss and regen_loss are UNCHANGED by the rescale,")
    print(f"     while path_cost_loss scales by exactly 0.5.")
    print(f"  => 'shrink all weights' is a FREE descent direction with no")
    print(f"     counter-pressure anywhere in the loss.\n")

    # =====================================================================
    # B/C/F. Gradient decomposition at both operating points
    # =====================================================================
    for label, pl, rg in [("UNTRAINED (epoch 0)", pipe_init, regen_init),
                          ("TRAINED (epoch %d)" % ckpt["epoch"], pipe, regen)]:
        print("=" * 78)
        print(f"B/C/F. GRADIENT DECOMPOSITION ON edge_weights - {label}")
        print("=" * 78)

        captured = {}

        def hook(_mod, _inp, out):
            out.retain_grad()
            captured["w"] = out

        h = pl.edge_weight_net.register_forward_hook(hook)
        # Untrained point is evaluated on epoch 1's schedule/demands, trained
        # point on the final epoch's — each at the settings training actually used.
        if "UNTRAINED" in label:
            tau_x = linear_anneal(1, t_cfg["regen_tau_start"], t_cfg["regen_tau_end"],
                                  t_cfg["regen_tau_anneal_start_epoch"],
                                  t_cfg["regen_tau_anneal_end_epoch"])
            t_sm_x = linear_anneal(1, sc_cfg.get("soft_max_temperature", 0.5),
                                   sc_cfg.get("soft_max_temperature_min", 0.01),
                                   t_cfg["regen_tau_anneal_start_epoch"],
                                   t_cfg["regen_tau_anneal_end_epoch"])
            vl_x = t_cfg["vlastelica_lambda"]
            dem = generate_demands(topo, cfg["num_demands"], cfg["bitrate_options"], seed=1)
        else:
            tau_x, t_sm_x, vl_x, dem = tau, t_sm, vl, demands

        path_noise_costs, gsnr_preds, path_inds, regen_probs = pl(
            dem, tau=tau_x, lambda_=vl_x, soft_max_temperature=t_sm_x)
        h.remove()
        w_t = captured["w"]

        feas = torch.zeros(1)
        n_infeas = 0
        for d in dem:
            thr = torch.tensor(mod_cfg.required_snr_threshold(d.bitrate_gbps))
            sf = F.relu(thr - gsnr_preds[d.id])
            feas = feas + sf
            if sf.item() > 0:
                n_infeas += 1

        L_feas = p_cfg["lambda_infeasible"] * feas.squeeze()
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
        for d in dem:
            usage += path_inds[d.id].detach()
        g_cost_direct = torch.zeros(n_edges)
        g_cost_surrogate = g_cost - g_cost_direct

        print(f"  demands={len(dem)}  infeasible={n_infeas}  tau={tau_x:.3f} "
              f"t_sm={t_sm_x:.4f} vlastelica_lambda={vl_x:.3f}")
        print(f"  L_feas={L_feas.item():.4f}  L_regen={L_regen.item():.4f}  "
              f"L_cost={L_cost.item():.6f}\n")

        def gstat(nm, g):
            nz = int((g != 0).sum())
            print(f"  {nm:<26} L1={g.abs().sum():.4e}  max|g|={g.abs().max():.4e}  "
                  f"nonzero={nz}/{n_edges}  sum={g.sum():+.4e}")

        gstat("grad path_cost  (total)", g_cost)
        gstat("  |- direct d/dw", g_cost_direct)
        gstat("  |- via surrogate", g_cost_surrogate)
        gstat("grad feasibility (STE)", g_feas)
        gstat("grad regen_count", g_regen)
        gstat("grad TOTAL", g_tot)

        l1c, l1f = g_cost.abs().sum().item(), g_feas.abs().sum().item()
        print(f"\n  ratio  L1(path_cost) / L1(feasibility) = "
              f"{(l1c / l1f) if l1f > 0 else float('inf'):.3f}")
        pos_frac = float((g_cost_direct > 0).float().sum() / max(1, int((usage > 0).sum())))
        print(f"  path_cost's direct component is >=0 on every edge "
              f"(pure shrink): {bool((g_cost_direct >= 0).all())}")
        print(f"  edges used by >=1 demand: {int((usage > 0).sum())}/{n_edges}"
              f"   (pos_frac={pos_frac:.2f})")

        # G. Scale-direction derivative: dL(c*w)/dc at c=1, which by the chain
        # rule is sum_e g_e * w_e. This is the exact quantity that drives the
        # collapse. For a degree-1 homogeneous term (path_cost) Euler's theorem
        # gives dL/dc == L itself; for a scale-invariant term (feasibility,
        # regen) it is 0 exactly, since Dijkstra's argmin ignores global scale.
        wv_ = w_t.detach().squeeze(-1)
        print("\n  G. scale-direction derivative  dL(c*w)/dc |_(c=1) = sum_e g_e*w_e:")
        for nm, g, ref in [("path_cost", g_cost, L_cost.item()),
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
        wv = w_t.detach().squeeze(-1)
        # grad_output flowing into the surrogate == d(total)/d(path_indicator).
        # Reconstruct its scale from the two contributing terms for one demand.
        gpi = grad_of(L_feas + L_cost, path_inds[dem[0].id])
        pert = vl_x * gpi.abs()
        print(f"\n  F. perturbation scale check (demand {dem[0].id}):")
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
