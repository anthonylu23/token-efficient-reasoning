#!/usr/bin/env python3
"""Wait for the frozen service to succeed, then run supplemental CPU-only analysis."""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

from token_efficient_reasoning.budget_analysis import analyze


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    while True:
        result = subprocess.run(
            [
                "systemctl",
                "--user",
                "show",
                args.unit,
                "-p",
                "ActiveState",
                "-p",
                "SubState",
                "-p",
                "Result",
            ],
            text=True,
            capture_output=True,
            check=True,
        )
        status = dict(line.split("=", 1) for line in result.stdout.splitlines())
        if status.get("Result") != "success" or status.get("ActiveState") == "failed":
            raise RuntimeError(f"M3 service did not succeed; preserving artifacts: {status}")
        if status.get("ActiveState") == "active" and status.get("SubState") == "exited":
            break
        if status.get("ActiveState") not in {"active", "activating"}:
            raise RuntimeError(f"M3 service stopped before confirmed completion: {status}")
        time.sleep(60)
    result = analyze(args.source_root, "primary", args.output_dir)
    print(f"Completed supplemental primary analysis: {len(result['rows'])} cells", flush=True)


if __name__ == "__main__":
    main()
