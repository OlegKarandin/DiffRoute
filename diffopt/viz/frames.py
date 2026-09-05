"""Streaming writer for the training-trajectory frame file.

One file per run. Every number in a frame is measured on the DEPLOYED (hard)
allocation: routes and cuts from `hard["alloc"]`, margins from
`hard["gsnr_preds"]`. The soft training pass is never mixed in — that is the
exact mismatch `hard_rollout`'s `gsnr_preds` was added to remove.

Streaming, not buffered. `append()` writes one JSON object per line to a
`frames.jsonl` sidecar and flushes, mirroring train.py's per-epoch
`f.flush()`, so a crashed 300-epoch run leaves a readable partial file the
way `e2e_train_log.csv` does. `close()` reads the sidecar back plus the
training CSV and assembles the single `frames.json`. This is also why
`stats` needs no accumulation here: spec section 4 specifies it as
`e2e_train_log.csv` verbatim.

The delta encoding is EXACT, not lossy. A lightpath's `seg_gsnr_db`,
`gsnr_db` and `margin_db` are a pure function of its own `segs` and
`cut_idx` (docs/architecture/invariants.md, "Physics layer"), so an entry
whose route and cuts are unchanged carries forward bit-identically.

Restoration seam (spec section 4.2): `(demand, scenario)` keying and the
`per_scenario`/`deployed`/`binding` site fields ship now, at |S| = 1, inert.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:  # pragma: no cover
    from diffopt.demands import Demand
    from diffopt.modulation import ModulationConfig
    from diffopt.topology import Topology

FORMAT_VERSION = 1

# The one scenario that exists today. Stage IV turns this into a list.
NOMINAL = "nominal"

# Hard decisions are exactly 0/1, so any threshold in (0, 1) works; 0.5 is
# what train.py's site_mask already uses.
_CUT = 0.5


def worst_chunk_gsnr_db(seg_gsnr_db: List[float], cut_idx: List[int]) -> float:
    """End-to-end GSNR: the WORST chunk, in dB.

    Noise adds WITHIN a chunk and never across a cut — a regenerator
    rebuilds the signal — so a chunk's noise is `sum_k 10^(-g_k/10)` over its
    own segments, and the path fails at whichever chunk is worst. This is
    `SegmentCombiner`'s chunk-max-noise semantics reproduced at hard
    decisions, where the DP fold's expectation collapses to exactly this.

    Boundary `k` sits between `seg[k]` and `seg[k+1]`, so `cut_idx` splits
    the segment list at `k + 1`.
    """
    if not seg_gsnr_db:
        return float("nan")
    cuts = sorted(set(cut_idx))
    bounds = [0] + [k + 1 for k in cuts] + [len(seg_gsnr_db)]
    worst_noise = 0.0
    for lo, hi in zip(bounds, bounds[1:]):
        if hi <= lo:
            continue
        noise = sum(10.0 ** (-g / 10.0) for g in seg_gsnr_db[lo:hi])
        worst_noise = max(worst_noise, noise)
    return -10.0 * math.log10(worst_noise)


def replay_frames(doc: dict) -> Dict[int, Dict[int, dict]]:
    """Reconstruct every epoch's full lightpath set from a keyframe+delta
    document: `epoch -> demand id -> lightpath entry`.

    This is the reference implementation of the reconstruction the viewer
    performs, and what
    tests/test_viz_frames.py::test_delta_replay_equals_a_keyframe_only_dump
    checks section 4.1's exactness claim against.
    """
    out: Dict[int, Dict[int, dict]] = {}
    current: Dict[int, dict] = {}
    for frame in doc["frames"]:
        if frame["keyframe"]:
            current = {}
        for lp in frame["lightpaths"]:
            current[lp["d"]] = lp
        out[frame["epoch"]] = dict(current)
    return out


class FrameWriter:
    """Same lifetime as train.py's CSV writers: constructed before the epoch
    loop, `append()`ed once per epoch, `close()`d after it."""

    def __init__(
        self,
        path: Path,
        *,
        topology: "Topology",
        demands: List["Demand"],
        cfg: dict,
        modulation_config: "ModulationConfig",
        every: int = 1,
        keyframe_every: int = 50,
    ) -> None:
        from diffopt.traffic import traffic_matrix_checksum
        from diffopt.viz.layout import frozen_layout

        if every < 1:
            raise ValueError(f"viz.every must be >= 1, got {every}")
        if keyframe_every < 1:
            raise ValueError(f"viz.keyframe_every must be >= 1, got {keyframe_every}")

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.sidecar = self.path.with_suffix(".jsonl")
        self.every = every
        self.keyframe_every = keyframe_every

        margin_db = float(cfg["constraint"]["margin_db"])
        self._thresholds = {
            d.id: float(modulation_config.required_snr_threshold(d.bitrate_gbps))
            for d in demands
        }

        self._header = {
            "format_version": FORMAT_VERSION,
            "run": {
                "config_path": cfg.get("_config_path", ""),
                "topology_path": cfg["topology"],
                "scenario": cfg["traffic"]["scenario"],
                "traffic_seed": int(cfg["traffic"]["seed"]),
                "traffic_checksum": traffic_matrix_checksum(demands),
                "margin_db": margin_db,
                "epochs_e2e": int(cfg["training"]["epochs_e2e"]),
                "selected_epoch": None,          # stamped by close()
            },
            "topology": {
                "num_nodes": topology.num_nodes,
                "edges": [
                    {"id": i, "src": e.src, "dst": e.dst, "km": float(e.length_km)}
                    for i, e in enumerate(topology.undirected_edges)
                ],
                "layout": frozen_layout(topology),
                "regen_candidates": list(topology.regen_candidate_nodes),
            },
            "failure_scenarios": [NOMINAL],
            "demands": [
                {
                    "id": d.id, "src": d.src, "dst": d.dst,
                    "bitrate_gbps": float(d.bitrate_gbps),
                    "threshold_db": round(self._thresholds[d.id], 4),
                    "bar_db": round(self._thresholds[d.id] + margin_db, 4),
                }
                for d in demands
            ],
        }

        self._prev: Dict[int, dict] = {}
        self._dumped = 0
        self._fh = open(self.sidecar, "w", encoding="utf-8", newline="\n")
        self._fh.write(json.dumps(self._header) + "\n")
        self._fh.flush()

    # -- per epoch ---------------------------------------------------------

    def append(self, epoch: int, hard: dict) -> None:
        """Build this epoch's frame from the DEPLOYED rollout and write it."""
        if epoch % self.every != 0:
            return
        alloc = hard["alloc"]
        keyframe = self._dumped % self.keyframe_every == 0
        self._dumped += 1

        entries = self._lightpaths(alloc, hard["gsnr_preds"])
        if keyframe:
            emitted = list(entries.values())
        else:
            emitted = [
                lp for did, lp in entries.items()
                # Only `segs` and `cut_idx` decide: everything else is a pure
                # function of them (section 4.1), so comparing them is
                # comparing the whole entry.
                if self._prev.get(did, {}).get("segs") != lp["segs"]
                or self._prev.get(did, {}).get("cut_idx") != lp["cut_idx"]
            ]
        self._prev = entries

        self._fh.write(json.dumps({
            "epoch": int(epoch),
            "keyframe": keyframe,
            "sites": self._sites(alloc),
            "lightpaths": emitted,
        }) + "\n")
        self._fh.flush()

    def _lightpaths(self, alloc, gsnr_preds) -> Dict[int, dict]:
        out: Dict[int, dict] = {}
        for row, did in enumerate(alloc.demand_ids):
            n = int(alloc.num_segments[row])
            cut_mask = alloc.a[row, : max(n - 1, 0)] > _CUT
            cut_idx = cut_mask.nonzero().flatten().tolist()
            cut_nodes = alloc.boundary_node_ids[row, : max(n - 1, 0)][cut_mask].tolist()
            seg_gsnr = [round(float(v), 4) for v in alloc.seg_gsnr_db[row, :n]]
            gsnr = float(gsnr_preds[did])
            out[did] = {
                "d": int(did),
                "s": NOMINAL,
                "segs": [list(map(int, g)) for g in alloc.segment_edge_ids[did]],
                "cut_idx": [int(k) for k in cut_idx],
                "cut_nodes": [int(nd) for nd in cut_nodes],
                "seg_gsnr_db": seg_gsnr,
                "seg_km": [round(float(v), 2) for v in alloc.seg_km_matrix[row, :n]],
                "gsnr_db": round(gsnr, 4),
                "margin_db": round(gsnr - self._thresholds[did], 4),
            }
        return out

    def _sites(self, alloc) -> Dict[str, dict]:
        """Bounded by num_nodes, so always written in full.

        `deployed` and `binding` are the Stage IV seam: at |S| = 1 they are
        `per_scenario["nominal"]` and `"nominal"`. At Stage IV `deployed`
        becomes `max_s` and `binding` names the scenario that sets it, which
        is what lets the map answer "node 83 has 4 devices BECAUSE edge 37
        fails" — invisible in any per-scenario view.
        """
        per_node = alloc.alloc_by_node.sum(dim=0)
        sites: Dict[str, dict] = {}
        for node in (per_node > _CUT).nonzero().flatten().tolist():
            count = int(round(float(per_node[node])))
            sites[str(int(node))] = {
                "per_scenario": {NOMINAL: count},
                "deployed": count,
                "binding": NOMINAL,
            }
        return sites

    # -- teardown ----------------------------------------------------------

    def close(self, selected_epoch: Optional[int], stats_csv: Path) -> None:
        """Assemble frames.json from the sidecar plus the training CSV."""
        self._fh.close()
        lines = self.sidecar.read_text(encoding="utf-8").splitlines()
        header = json.loads(lines[0])
        header["run"]["selected_epoch"] = (
            None if selected_epoch is None else int(selected_epoch)
        )
        header["stats"] = _read_stats(Path(stats_csv))
        header["frames"] = [json.loads(line) for line in lines[1:] if line.strip()]
        self.path.write_text(json.dumps(header), encoding="utf-8")

    def __del__(self) -> None:  # pragma: no cover - best-effort
        try:
            if not self._fh.closed:
                self._fh.close()
        except Exception:
            pass


def _read_stats(stats_csv: Path) -> dict:
    """e2e_train_log.csv verbatim, at FULL epoch resolution.

    Deliberately not delta-encoded and deliberately not subsampled by
    `every`: the stats panel's curves must not develop gaps the map happens
    to have (spec section 4).
    """
    if not stats_csv.exists():
        return {"columns": [], "rows": []}
    with open(stats_csv, newline="") as fh:
        reader = csv.reader(fh)
        columns = next(reader, [])
        rows = [[_num(v) for v in row] for row in reader if row]
    return {"columns": columns, "rows": rows}


def _num(v: str):
    """int where the CSV wrote an int, float where it wrote a float, and the
    raw string for the two boolean columns (`lookahead`, `route_context`)."""
    try:
        return int(v)
    except ValueError:
        pass
    try:
        f = float(v)
    except ValueError:
        return v
    return None if math.isnan(f) else f
