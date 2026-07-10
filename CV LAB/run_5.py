"""Unified entry point for Experiment 5 — OCR + Image Captioning.

Usage:
    python run_5.py run     Launch interactive OpenCV GUI
    python run_5.py all     Launch interactive OpenCV GUI
    python run_5.py steps   List available commands

Requires a display (OpenCV GUI windows).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

DIR = Path(__file__).parent
PY = sys.executable
SCRIPT = DIR / "experiment_5_ocr_captioning.py"


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="OCR + Captioning -- unified runner")
    parser.add_argument("command", nargs="?", default="steps", help="run | all | steps")
    args = parser.parse_args()

    if args.command == "steps":
        print("Available commands:")
        print("  python run_5.py run     Launch interactive OpenCV GUI")
        print("  python run_5.py all     Launch interactive OpenCV GUI")
        return

    if args.command in ("run", "all"):
        result = subprocess.run([PY, str(SCRIPT)], cwd=str(DIR))
        if result.returncode != 0:
            print(f"\n[!] Experiment exited with code {result.returncode}")
        return

    print(f"Unknown command: {args.command}")
    print("Run 'python run_5.py steps' for available commands.")
    sys.exit(1)


if __name__ == "__main__":
    main()
