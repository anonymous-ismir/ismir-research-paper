"""
Compatibility entrypoint.

The main runner lives in `backend/Evaluation/evaluation.py`.
"""

from __future__ import annotations

import os
import sys


def _add_backend_to_path() -> None:
    backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if backend_dir not in sys.path:
        sys.path.append(backend_dir)


_add_backend_to_path()


def main() -> None:
    from Evaluation.evaluation import main as runner_main

    runner_main()


if __name__ == "__main__":
    main()
