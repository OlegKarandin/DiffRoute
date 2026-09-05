"""Frame-format tests. Pure helpers plus a fake-`hard`-dict writer run —
no real pipeline, no checkpoint (tests/test_diagnose_scripts.py's style)."""
import json
from pathlib import Path

import pytest
import torch

from diffopt.demands import Demand
from diffopt.modulation import ModulationConfig
from diffopt.viz.frames import FrameWriter, replay_frames, worst_chunk_gsnr_db

BASE = Path(__file__).parent.parent
MODULATION_FORMATS_PATH = BASE / "configs/modulation_formats.yaml"


# ---------------------------------------------------------------------------
# worst_chunk_gsnr_db — the physics the strip draws
# ---------------------------------------------------------------------------

def test_worst_chunk_reproduces_the_spec_worked_example():
    """Spec section 4: three segments, one cut at boundary 0. Chunk 2 is
    two 16.62 dB segments: 10^(-1.662)*2 = 0.04355 -> 13.61 dB. End-to-end
    is the WORST chunk, min(18.20, 13.61)."""
    assert worst_chunk_gsnr_db([18.20, 16.62, 16.62], [0]) == pytest.approx(13.61, abs=0.01)


def test_worst_chunk_with_no_cuts_sums_every_segment():
    """One chunk: noise adds across the whole path."""
    assert worst_chunk_gsnr_db([16.62, 16.62], []) == pytest.approx(13.61, abs=0.01)


def test_worst_chunk_of_a_single_segment_is_that_segment():
    assert worst_chunk_gsnr_db([21.5], []) == pytest.approx(21.5)


def test_a_cut_never_lowers_the_result():
    """'Regen helps' at the strip's own arithmetic: cutting splits one chunk
    into two no-larger pieces (invariants.md, Segment combiner)."""
    segs = [20.0, 19.0, 21.0, 18.0]
    assert worst_chunk_gsnr_db(segs, [1]) >= worst_chunk_gsnr_db(segs, [])


# ---------------------------------------------------------------------------
# FrameWriter — a fake `hard` dict, so no pipeline is needed
# ---------------------------------------------------------------------------

def _mod_cfg():
    return ModulationConfig.from_yaml(str(MODULATION_FORMATS_PATH))


def _cfg():
    return {
        "topology": "configs/topology/german_17.json",
        "traffic": {"scenario": "stress", "seed": 0},
        "constraint": {"margin_db": 0.5},
        "training": {"epochs_e2e": 3},
    }


class _FakeAlloc:
    """Stand-in for a hard AllocationOutputs. Two demands: d0 has three
    segments and (optionally) a cut at boundary 0; d1 has one segment."""

    def __init__(self, *, d0_cut: bool, d0_route):
        self.demand_ids = [0, 1]
        self.segment_edge_ids = {0: d0_route, 1: [[2]]}
        self.num_segments = torch.tensor([3, 1])
        self.a = torch.tensor([[1.0 if d0_cut else 0.0, 0.0], [0.0, 0.0]])
        self.boundary_node_ids = torch.tensor([[3, 5], [-1, -1]])
        self.seg_gsnr_db = torch.tensor([[18.20, 16.62, 16.62], [30.0, 0.0, 0.0]])
        self.seg_km_matrix = torch.tensor([[412.0, 194.0, 194.0], [50.0, 0.0, 0.0]])
        n = 17
        self.alloc_by_node = torch.zeros(2, n)
        if d0_cut:
            self.alloc_by_node[0, 3] = 1.0


def _hard(*, d0_cut, d0_route=((0,), (1,), (4,))):
    alloc = _FakeAlloc(d0_cut=d0_cut, d0_route=[list(g) for g in d0_route])
    g0 = worst_chunk_gsnr_db([18.20, 16.62, 16.62], [0] if d0_cut else [])
    return {
        "alloc": alloc,
        "gsnr_preds": {0: torch.tensor(g0), 1: torch.tensor(30.0)},
        "alloc_by_node": alloc.alloc_by_node,
    }


def _demands():
    return [
        Demand(id=0, src=0, dst=4, bitrate_gbps=800.0),
        Demand(id=1, src=1, dst=2, bitrate_gbps=300.0),
    ]


def _write(tmp_path, sequence, **kw):
    """Run a FrameWriter over `sequence` (a list of `hard` dicts) and return
    the parsed document."""
    from diffopt.topology import load_topology

    topo = load_topology(
        str(BASE / "configs/topology/german_17.json"), str(MODULATION_FORMATS_PATH),
    )
    tmp_path.mkdir(parents=True, exist_ok=True)
    stats_csv = tmp_path / "e2e_train_log.csv"
    stats_csv.write_text(
        "epoch,total_loss\n"
        + "".join(f"{i + 1},{47.5 - i:.2f}\n" for i in range(len(sequence))),
        newline="",
    )
    out = tmp_path / "frames.json"
    w = FrameWriter(
        out, topology=topo, demands=_demands(), cfg=_cfg(),
        modulation_config=_mod_cfg(), **kw,
    )
    for i, hard in enumerate(sequence, start=1):
        w.append(i, hard)
    w.close(selected_epoch=len(sequence), stats_csv=stats_csv)
    return json.loads(out.read_text())


def test_delta_replay_equals_a_keyframe_only_dump(tmp_path):
    """Spec section 4.1's exactness claim. Replaying deltas must reproduce,
    field for field, what a keyframe-every-epoch dump contains."""
    seq = [
        _hard(d0_cut=False),
        _hard(d0_cut=False),                       # unchanged -> delta is empty
        _hard(d0_cut=True),                        # cut appears
        _hard(d0_cut=True, d0_route=((0,), (3,), (4,))),  # reroute
    ]
    delta = _write(tmp_path / "a", seq, keyframe_every=50)
    full = _write(tmp_path / "b", seq, keyframe_every=1)

    assert replay_frames(delta) == replay_frames(full)
    # And the delta encoding actually saved something.
    n_delta = sum(len(f["lightpaths"]) for f in delta["frames"])
    n_full = sum(len(f["lightpaths"]) for f in full["frames"])
    assert n_delta < n_full


def test_an_unchanged_epoch_writes_no_lightpath_entries(tmp_path):
    doc = _write(tmp_path, [_hard(d0_cut=False), _hard(d0_cut=False)], keyframe_every=50)
    assert doc["frames"][0]["keyframe"] is True
    assert len(doc["frames"][0]["lightpaths"]) == 2
    assert doc["frames"][1]["keyframe"] is False
    assert doc["frames"][1]["lightpaths"] == []


def test_every_lightpath_gsnr_equals_its_own_worst_chunk(tmp_path):
    """Spec section 6.7: a mismatch means the DP fold and the deployed
    rollout disagree about chunk noise, which is worth failing on."""
    doc = _write(tmp_path, [_hard(d0_cut=True), _hard(d0_cut=False)], keyframe_every=1)
    for frame in doc["frames"]:
        for lp in frame["lightpaths"]:
            assert lp["gsnr_db"] == pytest.approx(
                worst_chunk_gsnr_db(lp["seg_gsnr_db"], lp["cut_idx"]), abs=1e-3,
            )


def test_margin_is_against_the_bare_threshold_and_the_bar_is_above_it(tmp_path):
    """Spec section 6.1: the two lines are different things. 0 dB is physics;
    run.margin_db is a surrogate-error buffer."""
    doc = _write(tmp_path, [_hard(d0_cut=True)], keyframe_every=1)
    assert doc["run"]["margin_db"] == 0.5
    by_id = {d["id"]: d for d in doc["demands"]}
    assert by_id[0]["threshold_db"] == pytest.approx(15.1)   # 800G
    assert by_id[0]["bar_db"] == pytest.approx(15.6)
    lp = next(l for l in doc["frames"][0]["lightpaths"] if l["d"] == 0)
    assert lp["margin_db"] == pytest.approx(lp["gsnr_db"] - 15.1, abs=1e-3)


def test_cut_idx_and_cut_nodes_agree(tmp_path):
    """cut_idx is load-bearing, cut_nodes is display convenience — but they
    must describe the same cuts."""
    doc = _write(tmp_path, [_hard(d0_cut=True)], keyframe_every=1)
    lp = next(l for l in doc["frames"][0]["lightpaths"] if l["d"] == 0)
    assert lp["cut_idx"] == [0]
    assert lp["cut_nodes"] == [3]
    assert len(lp["segs"]) == len(lp["seg_gsnr_db"]) == len(lp["seg_km"]) == 3


def test_restoration_seam_ships_inert(tmp_path):
    """At |S| = 1 the seam must be present and consistent, not absent.
    Asserting it now is what stops the fields being dropped as unused."""
    doc = _write(tmp_path, [_hard(d0_cut=True)], keyframe_every=1)
    assert doc["failure_scenarios"] == ["nominal"]
    for frame in doc["frames"]:
        for lp in frame["lightpaths"]:
            assert lp["s"] == "nominal"
        for site in frame["sites"].values():
            assert site["deployed"] == site["per_scenario"]["nominal"]
            assert site["binding"] == "nominal"


def test_static_header_carries_the_run_identity(tmp_path):
    doc = _write(tmp_path, [_hard(d0_cut=True)], keyframe_every=1)
    assert doc["format_version"] == 1
    assert doc["run"]["scenario"] == "stress"
    assert doc["run"]["traffic_seed"] == 0
    assert doc["run"]["selected_epoch"] == 1
    assert len(doc["run"]["traffic_checksum"]) == 16
    assert doc["topology"]["num_nodes"] == 17
    assert len(doc["topology"]["layout"]) == 17
    assert doc["topology"]["edges"][0].keys() >= {"id", "src", "dst", "km"}
    assert doc["stats"]["columns"] == ["epoch", "total_loss"]
    assert doc["stats"]["rows"][0] == [1, 47.5]


def test_every_n_subsamples_frames_but_not_stats(tmp_path):
    """Spec section 4: stats stay at FULL epoch resolution even when
    `--every N` subsamples frames — the curves must not develop gaps the
    map happens to have."""
    seq = [_hard(d0_cut=i % 2 == 0) for i in range(4)]
    doc = _write(tmp_path / "a", seq, every=2, keyframe_every=50)
    assert [f["epoch"] for f in doc["frames"]] == [2, 4]
    assert len(doc["stats"]["rows"]) == 4
    assert doc["frames"][0]["keyframe"] is True   # first DUMPED frame is a keyframe


def test_a_crashed_run_leaves_a_readable_sidecar(tmp_path):
    """close() never being reached must still leave per-epoch data on disk,
    the way e2e_train_log.csv does."""
    from diffopt.topology import load_topology

    topo = load_topology(
        str(BASE / "configs/topology/german_17.json"), str(MODULATION_FORMATS_PATH),
    )
    out = tmp_path / "frames.json"
    w = FrameWriter(
        out, topology=topo, demands=_demands(), cfg=_cfg(),
        modulation_config=_mod_cfg(),
    )
    w.append(1, _hard(d0_cut=True))
    del w  # no close()
    lines = (tmp_path / "frames.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2                      # header + one frame
    assert json.loads(lines[1])["epoch"] == 1


# ---------------------------------------------------------------------------
# scripts/dump_frames.py — pure helpers (conftest.py already puts scripts/
# on sys.path)
# ---------------------------------------------------------------------------

def test_dump_frames_resolves_the_output_path_from_the_config():
    from dump_frames import resolve_out_path  # noqa: E402

    cfg = {"log_dir": "logs/constrained_stress"}
    assert resolve_out_path(cfg, None) == Path("logs/constrained_stress/frames.json")
    assert resolve_out_path(cfg, "x/y.json") == Path("x/y.json")


def test_dump_frames_stamps_the_checkpoint_epoch_as_selected():
    """A post-hoc dump has exactly one frame, and it IS the selected
    checkpoint — so run.selected_epoch must name that epoch, not None."""
    from dump_frames import selected_epoch_of  # noqa: E402

    assert selected_epoch_of({"epoch": 201}) == 201
    assert selected_epoch_of({}) is None
    assert selected_epoch_of(None) is None
