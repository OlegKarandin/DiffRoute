"""The viewer build step: inlining frames.json into the template.

Pure string work — no browser, no artifact. The page's own behaviour is
verified by the self-check it runs on load (see viewer.html) and by manual
inspection in Task 12.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from build_viewer import build_viewer, PLACEHOLDER  # noqa: E402

TEMPLATE = Path(__file__).parent.parent / "diffopt/viz/viewer.html"


def test_template_has_exactly_one_placeholder():
    assert TEMPLATE.read_text(encoding="utf-8").count(PLACEHOLDER) == 1


def test_template_is_artifact_ready():
    """The platform supplies the document skeleton; a page that brings its
    own is malformed once wrapped."""
    text = TEMPLATE.read_text(encoding="utf-8").lower()
    for tag in ("<!doctype", "<html", "<head", "<body"):
        assert tag not in text
    assert "<title>" in text


def test_template_never_fetches():
    """fetch/XHR are blocked in the artifact sandbox, so the data must be
    inlined rather than loaded."""
    text = TEMPLATE.read_text(encoding="utf-8")
    assert "fetch(" not in text
    assert "XMLHttpRequest" not in text


def test_build_substitutes_the_data_and_leaves_no_placeholder():
    out = build_viewer(f"A{PLACEHOLDER}B", '{"format_version": 1}')
    assert out == 'A{"format_version": 1}B'
    assert PLACEHOLDER not in out


def test_build_escapes_a_closing_script_tag_inside_the_json():
    """A string in the data must never be able to close the script tag."""
    payload = json.dumps({"config_path": "a</script>b"})
    out = build_viewer(PLACEHOLDER, payload)
    assert "</script>" not in out
    assert r"<\/script>" in out
    # Still valid JSON to a browser: <\/ is a legal JSON escape for /.
    assert json.loads(out)["config_path"] == "a</script>b"


def test_build_rejects_a_template_without_the_placeholder():
    with pytest.raises(ValueError, match="placeholder"):
        build_viewer("no slot here", "{}")
