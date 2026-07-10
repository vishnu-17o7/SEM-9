"""Unified entry point for SMS Spam ML Compare.

Usage:
    python run.py <command> [options]

Commands:
    train     Train and compare 8 models
    report    Generate HTML report
    web       Launch web UI (port 5004)
    all       Run full pipeline (train + report)
    steps     List available commands
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

DIR = Path(__file__).parent
PY = sys.executable

COMMANDS: dict[str, list[list[str]]] = {
    "train": [["model_comparison.py"]],
    "report": [["report.py"]],
}

WEB_SCRIPT: str | None = "app.py"


def run_scripts(steps: list[list[str]]) -> None:
    """Run a list of [script, args...] in sequence."""
    for step in steps:
        script = step[0]
        args = step[1:]
        print(f"\n{'=' * 60}")
        print(f"  >> {script} {' '.join(args)}".strip())
        print(f"{'=' * 60}\n")
        result = subprocess.run([PY, str(DIR / script)] + args, cwd=str(DIR))
        if result.returncode != 0:
            print(f"\n  [!] {script} failed (exit {result.returncode})")
            sys.exit(1)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="SMS Spam ML Compare -- unified runner")
    parser.add_argument("command", nargs="?", default="steps",
                        help="train | report | web | all | steps")
    parser.add_argument("cmd_args", nargs="*", help="Extra args for web")
    args = parser.parse_args()

    cmd = args.command

    if cmd == "steps":
        available = list(COMMANDS.keys())
        if WEB_SCRIPT:
            available.append("web")
        available.append("all")
        print("Available commands:")
        for c in available:
            print(f"  python run.py {c}")
        return

    if cmd == "all":
        steps: list[list[str]] = []
        for step_list in COMMANDS.values():
            steps.extend(step_list)
        run_scripts(steps)
        print("\n  [OK] Pipeline complete!")
        return

    if cmd in COMMANDS:
        run_scripts(COMMANDS[cmd])
        return

    if cmd == "web" and WEB_SCRIPT:
        run_scripts([[WEB_SCRIPT] + args.cmd_args])
        return

    print(f"Unknown command: {cmd}")
    print("Run 'python run.py steps' for available commands.")
    sys.exit(1)


if __name__ == "__main__":
    main()
