import sys
from pathlib import Path

_root = Path(__file__).parent.parent
_tests = Path(__file__).parent

for _p in (_root, _tests):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
