import sys
from pathlib import Path

# Ensure src is on sys.path when running from repo root or driver
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
# Repo root on path so `scripts.*` (namespace packages, no __init__.py) are importable in tests.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
