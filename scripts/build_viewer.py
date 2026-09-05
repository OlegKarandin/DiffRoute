"""Inline a frames.json into diffopt/viz/viewer.html -> one self-contained page.

The artifact sandbox blocks fetch/XHR and every non-allowlisted external
resource, so an external data file cannot be loaded (spec section 6). The
data is therefore substituted into the template at build time.

Usage:
    conda activate diffopt
    python scripts/build_viewer.py \
        --frames logs/constrained_stress/frames.json \
        --out build/trajectory_viewer.html
"""
from __future__ import annotations

import argparse
from pathlib import Path

PLACEHOLDER = "__FRAMES_JSON__"
TEMPLATE = Path(__file__).parent.parent / "diffopt/viz/viewer.html"


def build_viewer(template: str, frames_json: str) -> str:
    """Substitute `frames_json` into `template`, escaping any `</` so a
    string in the data can never close the <script> tag early. `<\\/` is a
    legal JSON escape for `/`, so the result still parses."""
    if PLACEHOLDER not in template:
        raise ValueError(f"template has no {PLACEHOLDER} placeholder")
    return template.replace(PLACEHOLDER, frames_json.replace("</", r"<\/"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--template", default=str(TEMPLATE))
    args = ap.parse_args()

    data = Path(args.frames).read_text(encoding="utf-8")
    page = build_viewer(Path(args.template).read_text(encoding="utf-8"), data)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    mb = len(page.encode("utf-8")) / 1e6
    print(f"{out}  ({mb:.2f} MB; the artifact cap is 16 MB)")
    if mb > 15.0:
        print("WARNING: close to the cap — consider `viz.every: 2` and re-dumping")


if __name__ == "__main__":
    main()
