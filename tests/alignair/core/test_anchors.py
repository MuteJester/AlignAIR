"""The conserved V and J anchors have one product definition (``alignair.core.anchors``): their codon length, the genes
they belong to, and their target and output keys. Every other module reads them from there."""
import ast
from pathlib import Path

import pytest

from alignair.core import anchors

pytestmark = pytest.mark.component

SRC = Path(__file__).resolve().parents[3] / "src"
DEFINITION = SRC / "alignair" / "core" / "anchors.py"
# what only the definition may spell: the codon length's name, and the anchors' target and output keys
NAMES = {"CODON", "ANCHORS", "ANCHOR_TARGETS", "ANCHOR_GENES", "ANCHOR_KEYS", "ANCHOR_LOGITS"}
KEYS = {"v_anchor", "j_anchor", "v_anchor_logits", "j_anchor_logits", "_anchor", "_anchor_logits"}


def test_the_anchors_are_defined_once():
    assert anchors.CODON == 3 and anchors.ANCHOR_GENES == ("v", "j")
    assert anchors.ANCHOR_KEYS == {"v": "v_anchor", "j": "j_anchor"}
    assert anchors.ANCHOR_LOGITS == {"v": "v_anchor_logits", "j": "j_anchor_logits"}


def test_an_anchor_can_start_only_where_a_whole_codon_fits_in_the_read():
    """Reads of 5, 3 and 2 bases in a window of 6: codon starts 0..2, 0, and none."""
    import torch
    read = torch.tensor([[1, 1, 1, 1, 1, 0], [1, 1, 1, 0, 0, 0], [1, 1, 0, 0, 0, 0]], dtype=torch.bool)
    assert anchors.codon_starts(read).tolist() == [[True, True, True, False, False, False],
                                                   [True, False, False, False, False, False], [False] * 6]
    assert read.tolist()[0] == [True] * 5 + [False]                  # the read mask itself is left as it was


def test_no_other_module_defines_the_codon_or_spells_an_anchor_key():
    found = []
    for path in sorted([*(SRC / "alignair").rglob("*.py"), *(SRC / "alignair_lab").rglob("*.py")]):
        if path == DEFINITION or "applicability" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(
                node, ast.AnnAssign) else []
            found += [f"{path.relative_to(SRC)}:{node.lineno} assigns {target.id}" for target in targets
                      if isinstance(target, ast.Name) and target.id in NAMES]
        found += [f"{path.relative_to(SRC)}:{node.lineno} spells {node.value!r}" for node in ast.walk(tree)
                  if isinstance(node, ast.Constant) and node.value in KEYS]
    assert not found, "\n".join(found)
