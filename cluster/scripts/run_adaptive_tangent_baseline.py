#!/usr/bin/env python3
"""Run the source-adaptive tangent-space CLIP arithmetic experiment."""

from __future__ import annotations

import sys

from run_baseline import main


if __name__ == "__main__":
    sys.argv[1:1] = ["--method", "adaptive_tangent_sequential"]
    raise SystemExit(main())
