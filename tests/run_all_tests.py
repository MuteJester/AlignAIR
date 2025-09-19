import sys
import pytest
from pathlib import Path

if __name__ == "__main__":
    tests_dir = Path(__file__).resolve().parent
    # Allow passing additional pytest args, e.g., -q or -k filters
    sys.exit(pytest.main([str(tests_dir)] + sys.argv[1:]))
