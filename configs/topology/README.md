# Topologies

The `*.json` files here are the topology artifacts the pipeline consumes. They
are committed, so nothing in this repo requires the `.dat` sources to run.

Each JSON is generated from a `.dat` source by `diffopt/topology_builder.py`,
which splits every link into balanced spans (see CLAUDE.md, "Topology"
constraints). Regenerating them requires the `.dat` files, which are **not
distributed here**:

| topology | nodes | source |
|---|---|---|
| `ind_132.json` | 132 | `IND_132.dat` — Karandin et al., *J. Opt. Commun. Netw.* **16**, H18-H26 (2024), https://github.com/OlegKarandin/jocn24-multi-fiber |
| `jp_70.json` | 69 | `JP_70.dat` — same source |
| `german_17.json` | 17 | `german_17.dat` |
| `eu_19.json` | 19 | `EU_19.dat` |

`ind_132` is the production topology. `german_17` is the legacy smoke-test
topology and is what the GNPy ground-truth test in
`tests/test_optical_bridge.py` was measured against.

Note `jp_70.json` has 69 nodes despite its name, which follows the source file.

If you obtain the `.dat` sources, place them at the repo root and run:

    python diffopt/topology_builder.py
