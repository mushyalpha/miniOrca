import sys
from pathlib import Path

_here = Path(__file__).parent         
_root = _here.parent                  

for _p in (_root, _here):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)
