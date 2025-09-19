import sys
from pathlib import Path

# Ensure src is on sys.path when running from repo root or driver
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
