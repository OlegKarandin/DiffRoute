"""Is the placement head actually learning, or just drifting uniformly?

Falsifiable null model: if NO node ever learns anything node-specific and
every logit merely slides in one direction by ~lr_regen per Adam step
(Adam's step size is ~lr whenever the gradient sign is constant, regardless
of magnitude), what regen_loss curve would that produce?

    logit(e)      = -lr_regen * e
    p(e)          = sigmoid(logit / tau(e))
    regen_loss(e) = num_nodes * p(e)

If the null model reproduces the real curve closely, the placement head made
no node-specific decision. On the pre-fix ind_132 run it matched to 3.49%
mean relative error over 20 epochs — see
docs/investigations/regen_placement_not_concentrating.md.

Also decomposes how much of the visible regen_loss movement came from tau
annealing rather than from learning: sigmoid(x/tau) moves dramatically as tau
shrinks even when x is static (87% of the pre-fix fall was tau, not learning).

Usage:
    conda activate diffopt
    python scripts/diagnose_logit_drift.py --log logs/e2e_ind132/e2e_train_log.csv
"""
import argparse, csv, math
from pathlib import Path
import yaml

from _common import add_common_args, schedule_at

ap = argparse.ArgumentParser()
add_common_args(ap, with_checkpoint=False, with_demands=False)
ap.add_argument("--log", default="logs/e2e_ind132/e2e_train_log.csv")
ap.add_argument("--num-nodes", type=int, default=132)
args = ap.parse_args()

cfg = yaml.safe_load(Path(args.config).read_text())
t_cfg = cfg["training"]
N = args.num_nodes
LR = t_cfg["lr_regen"]

rows = list(csv.DictReader(open(args.log)))

print(f"pure-drift model: logit(e) = -{LR} * e, no node-specific learning at all\n")
print(f"{'ep':>3} {'tau':>6} {'logit':>8} {'p_pred':>7} | "
      f"{'regen_loss':>10} {'predicted':>10} {'err':>7} | {'n_regen>0.5':>11}")
errs = []
for r in rows:
    e = int(r["epoch"])
    tau, _, _ = schedule_at(cfg, epoch=e)
    logit = -LR * e
    p = 1.0 / (1.0 + math.exp(-logit / tau))
    pred = N * p
    act = float(r["regen_loss"])
    errs.append(abs(pred - act) / act)
    print(f"{e:3d} {tau:6.3f} {logit:8.4f} {p:7.4f} | {act:10.3f} {pred:10.3f} "
          f"{pred-act:+7.3f} | {r['num_regen_soft']:>11}")

print(f"\nmean relative error of the pure-drift prediction: {100*sum(errs)/len(errs):.2f}%")

last = rows[-1]
n_ep = int(last["epoch"])
tau_end, _, _ = schedule_at(cfg, epoch=n_ep)

if "regen_logit_min" in last:
    lo, hi = float(last["regen_logit_min"]), float(last["regen_logit_max"])
    print(f"\nfinal raw logits (logged): [{lo:+.4f}, {hi:+.4f}]  "
          f"mean {float(last['regen_logit_mean']):+.4f}")
else:
    # older logs recorded only regen_loss; back-solve the mean logit from it
    pm = float(last["regen_loss"]) / N
    lo = hi = tau_end * math.log(pm / (1 - pm))
    print(f"\nlog predates raw-logit columns; back-solved mean logit "
          f"from regen_loss: {lo:+.4f}")

band = hi - lo
print(f"pure drift over {n_ep} epochs would give a single logit at "
      f"{-LR * n_ep:+.4f} (all nodes identical)")
print(f"observed spread across nodes: {band:.4f} "
      f"= {band / LR:.1f} Adam steps' worth")
if band < abs(LR * n_ep):
    print("=> logits are packed tighter than the distance they travelled: they "
          "moved as a BLOCK, no node-specific decision was made")
elif lo < 0 < hi:
    print("=> logits straddle zero: the head has SPLIT nodes into keep/drop "
          "populations, which is what selective placement looks like")
