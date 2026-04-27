"""
Read-only shim for importing from the main project without touching it.

After `import extensions._shim`, the following imports work from any
script under extensions/:

    from feature_extractor import extract_all_features, ALL_FEATURE_NAMES
    from data_loader import load_xpqrs, XPQRS_CLASSES

The shim never writes to or modifies anything in the main project.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_SRC_DIR   = os.path.join(_REPO_ROOT, "src")

for p in (_SRC_DIR, _REPO_ROOT):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)


REPO_ROOT = _REPO_ROOT
SRC_DIR   = _SRC_DIR

# Default location of the existing trained RF
LEGACY_RF_PATH = os.path.join(
    _REPO_ROOT, "results", "models", "xpqrs_random_forest.pkl"
)
# Default raw-waveform dataset
XPQRS_DATA_DIR = os.path.join(_REPO_ROOT, "dataset", "XPQRS")
