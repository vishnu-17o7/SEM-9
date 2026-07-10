"""Unified entry point for Experiment 11 — Fashion-MNIST CNN.

Usage:
    python run_11.py run [--epochs N] [--batch-size N] [--quick]
    python run_11.py all [--epochs N] [--batch-size N] [--quick]
    python run_11.py steps

Requires a display (OpenCV GUI windows + matplotlib).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

DIR = Path(__file__).parent
PY = sys.executable
SCRIPT = DIR / "experiment_11_fashion_mnist.py"


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Fashion-MNIST CNN -- unified runner")
    parser.add_argument("command", nargs="?", default="steps", help="run | all | steps")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs (default: 5)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size (default: 128)")
    parser.add_argument("--quick", action="store_true", help="Quick mode (fast training)")
    args = parser.parse_args()

    if args.command == "steps":
        print("Available commands:")
        print("  python run_11.py run                     Launch with training (5 epochs)")
        print("  python run_11.py run --epochs 10         Training with 10 epochs")
        print("  python run_11.py run --quick             Quick training mode")
        print("  python run_11.py all                     Same as run")
        return

    if args.command in ("run", "all"):
        cmd = [PY, str(SCRIPT), f"--epochs={args.epochs}", f"--batch-size={args.batch_size}"]
        if args.quick:
            cmd.append("--quick")
        result = subprocess.run(cmd, cwd=str(DIR))
        if result.returncode != 0:
            print(f"\n[!] Experiment exited with code {result.returncode}")
        return

    print(f"Unknown command: {args.command}")
    print("Run 'python run_11.py steps' for available commands.")
    sys.exit(1)


if __name__ == "__main__":
    main()
