"""Unified entry point for Experiment 7 — Custom Defect Detection (Faster R-CNN).

Usage:
    python run_7.py run [--epochs N] [--n-synth N]     Launch with training
    python run_7.py all [--epochs N] [--n-synth N]     Same as run
    python run_7.py steps                                List available commands

Requires a display (OpenCV GUI windows + matplotlib).
Delegates to program_7_defect_detection/main.py.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

DIR = Path(__file__).parent
PY = sys.executable
SCRIPT = DIR / "experiment_7_defect_detection.py"


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Defect Detection -- unified runner")
    parser.add_argument("command", nargs="?", default="steps", help="run | all | steps")
    parser.add_argument("--epochs", type=int, default=4, help="Training epochs (default: 4)")
    parser.add_argument("--n-synth", type=int, default=120, help="Synthetic samples (default: 120)")
    args = parser.parse_args()

    if args.command == "steps":
        print("Available commands:")
        print("  python run_7.py run              Launch with training (defaults: epochs=4, n-synth=120)")
        print("  python run_7.py run --epochs 8   Training with 8 epochs")
        print("  python run_7.py all              Same as run")
        return

    if args.command in ("run", "all"):
        cmd = [PY, str(SCRIPT), f"--epochs={args.epochs}", f"--n-synth={args.n_synth}"]
        result = subprocess.run(cmd, cwd=str(DIR))
        if result.returncode != 0:
            print(f"\n[!] Experiment exited with code {result.returncode}")
        return

    print(f"Unknown command: {args.command}")
    print("Run 'python run_7.py steps' for available commands.")
    sys.exit(1)


if __name__ == "__main__":
    main()
