"""SlmCortex package shim over a flat src layout."""

import importlib
from pathlib import Path

__path__ = [str(Path(__file__).resolve().parent)]
__version__ = "0.1.1"


def main(argv=None):
    return importlib.import_module("slmcortex.cli").main(argv)


def composer_main(argv=None):
    composer_argv = ["composer-app"]
    if argv:
        composer_argv.extend(argv)
    return main(composer_argv)


if __name__ == "__main__":
    raise SystemExit(main())
