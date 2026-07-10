"""Unified entry point for Experiment 12 — Small Object + Satellite Multi-label.

Usage:
    python run_12.py run [--epochs N] [--quick]     Launch with training
    python run_12.py all [--epochs N] [--quick]     Same as run
    python run_12.py steps                           List available commands

Requires a display (OpenCV GUI windows + matplotlib).
Delegates to program_12_smallobj_satellite/main.py.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

DIR = Path(__file__).parent
PY = sys.executable
SCRIPT = DIR / "experiment_12_smallobj_satellite.py"


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Small Object + Satellite -- unified runner")
    parser.add_argument("command", nargs="?", default="steps", help="run | all | steps")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs (default: 3)")
    parser.add_argument("--quick", action="store_true", help="Quick training mode")
    args = parser.parse_args()

    if args.command == "steps":
        print("Available commands:")
        print("  python run_12.py run                     Launch with training (3 epochs)")
        print("  python run_12.py run --epochs 5          Training with 5 epochs")
        print("  python run_12.py run --quick             Quick training mode")
        print("  python run_12.py all                     Same as run")
        return

    if args.command in ("run", "all"):
        cmd = [PY, str(SCRIPT), f"--epochs={args.epochs}"]
        if args.quick:
            cmd.append("--quick")
        result = subprocess.run(cmd, cwd=str(DIR))
        if result.returncode != 0:
            print(f"\n[!] Experiment exited with code {result.returncode}")
        return

    print(f"Unknown command: {args.command}")
    print("Run 'python run_12.py steps' for available commands.")
    sys.exit(1)


if __name__ == "__main__":
    main()
