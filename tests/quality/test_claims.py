"""Every model-card claim maps to a real named test (``tests.support.claims.CLAIM_TESTS``)."""
import ast

import pytest

from tests.support.claims import CLAIM_TESTS

pytestmark = pytest.mark.acceptance


@pytest.mark.parametrize("claim,nodeid", list(CLAIM_TESTS.items()))
def test_every_claim_points_at_a_real_test(claim, nodeid, repo_root):
    path, _, func = nodeid.partition("::")
    full = repo_root / path
    assert full.is_file(), f"claim {claim!r}: test file {path} does not exist"
    functions = {node.name for node in ast.walk(ast.parse(full.read_text())) if isinstance(node, ast.FunctionDef)}
    assert func in functions, f"claim {claim!r}: {func} not found in {path}"
