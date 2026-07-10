"""Unified entry point for Experiment 9 — Face Recognition (VGGFace2).

Usage:
    python run_9.py run [--mode detect|identify|verify] [--threshold F]
    python run_9.py all [--mode detect|identify|verify] [--threshold F]
    python run_9.py steps

Requires a display (OpenCV GUI windows).
Delegates to program_9_vggface2/main.py.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

DIR = Path(__file__).parent
PY = sys.executable
SCRIPT = DIR / "experiment_9_vggface2.py"


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Face Recognition -- unified runner")
    parser.add_argument("command", nargs="?", default="steps", help="run | all | steps")
    parser.add_argument("--mode", choices=["detect", "identify", "verify"], default="identify")
    parser.add_argument("--threshold", type=float, default=0.6)
    args = parser.parse_args()

    if args.command == "steps":
        print("Available commands:")
        print("  python run_9.py run                        Launch in identify mode")
        print("  python run_9.py run --mode verify          Launch in verify mode")
        print("  python run_9.py all                        Same as run")
        return

    if args.command in ("run", "all"):
        cmd = [PY, str(SCRIPT), f"--mode={args.mode}", f"--threshold={args.threshold}"]
        result = subprocess.run(cmd, cwd=str(DIR))
        if result.returncode != 0:
            print(f"\n[!] Experiment exited with code {result.returncode}")
        return

    print(f"Unknown command: {args.command}")
    print("Run 'python run_9.py steps' for available commands.")
    sys.exit(1)


if __name__ == "__main__":
    main()
