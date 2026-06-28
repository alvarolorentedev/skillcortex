"""SlmCortex package shim over a flat src layout."""

import importlib
from pathlib import Path

__path__ = [str(Path(__file__).resolve().parent)]
__version__ = "0.1.0"


def main(argv=None):
    return importlib.import_module("slmcortex.cli").main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
