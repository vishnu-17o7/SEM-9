"""Unified entry point for Experiment 10 — FaceNet Classification.

Usage:
    python run_10.py run [--classifier svm|knn] [--test-size F]
    python run_10.py all [--classifier svm|knn] [--test-size F]
    python run_10.py steps

Requires a display (OpenCV GUI windows + matplotlib).
Delegates to program_10_facenet/main.py.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

DIR = Path(__file__).parent
PY = sys.executable
SCRIPT = DIR / "experiment_10_facenet.py"


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="FaceNet Classification -- unified runner")
    parser.add_argument("command", nargs="?", default="steps", help="run | all | steps")
    parser.add_argument("--classifier", choices=["svm", "knn"], default="svm")
    parser.add_argument("--test-size", type=float, default=0.25)
    args = parser.parse_args()

    if args.command == "steps":
        print("Available commands:")
        print("  python run_10.py run                        Launch with SVM (default)")
        print("  python run_10.py run --classifier knn       Launch with KNN")
        print("  python run_10.py all                        Same as run")
        return

    if args.command in ("run", "all"):
        cmd = [PY, str(SCRIPT), f"--classifier={args.classifier}", f"--test-size={args.test_size}"]
        result = subprocess.run(cmd, cwd=str(DIR))
        if result.returncode != 0:
            print(f"\n[!] Experiment exited with code {result.returncode}")
        return

    print(f"Unknown command: {args.command}")
    print("Run 'python run_10.py steps' for available commands.")
    sys.exit(1)


if __name__ == "__main__":
    main()
