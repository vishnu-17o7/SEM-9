"""Unified entry point for Spam/Ham Watcher.

Usage:
    python run.py <command> [options]

Commands:
    train     Download data and train the spam classifier
    classify  Classify a text message (interactive)
    watch     Start Gmail IMAP watcher
    web       Launch FastAPI dashboard
    all       Run full pipeline (train)
    steps     List available commands
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

DIR = Path(__file__).parent
PY = sys.executable

COMMANDS: dict[str, list[list[str]]] = {
    "data": [["train_text_model.py", "--data-only"]],
    "train": [["train_text_model.py"]],
    "report": [["report.py"]],
}

EXTRA_COMMANDS: dict[str, str] = {
    "classify": "classify.py",
    "watch": "gmail_watcher.py",
    "web": "app.py",
}


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

    parser = argparse.ArgumentParser(description="Spam/Ham Watcher -- unified runner")
    parser.add_argument("command", nargs="?", default="steps",
                        help="data | train | report | audit | classify | watch | web | all | steps")
    parser.add_argument("--source", choices=["auto", "real", "synthetic"], default="auto")
    parser.add_argument("--input", type=Path, help="Optional local dataset or source archive")
    parser.add_argument("cmd_args", nargs="*", help="Extra args for classify/watch/web")
    args = parser.parse_args()

    cmd = args.command

    if cmd == "steps":
        available = list(COMMANDS.keys()) + list(EXTRA_COMMANDS.keys()) + ["all"]
        print("Available commands:")
        for c in available:
            print(f"  python run.py {c}")
        return

    if cmd == "all":
        steps: list[list[str]] = [["train_text_model.py", "--data-only", "--source", args.source], ["train_text_model.py"]]
        run_scripts(steps)
        run_scripts([[str(DIR.parents[1] / "CTI LAB" / "evaluation_guard.py"), "--project", str(DIR)]])
        run_scripts([["report.py"]])
        run_scripts([[str(DIR.parents[1] / "CTI LAB" / "report_badge.py"), "--project", str(DIR)]])
        print("\n  [OK] Pipeline complete!")
        return

    if cmd == "audit":
        run_scripts([[str(DIR.parents[1] / "CTI LAB" / "evaluation_guard.py"), "--project", str(DIR)]])
        return

    if cmd == "data":
        run_scripts([["train_text_model.py", "--data-only", "--source", args.source]])
        return

    if cmd in COMMANDS:
        run_scripts(COMMANDS[cmd])
        return

    if cmd in EXTRA_COMMANDS:
        run_scripts([[EXTRA_COMMANDS[cmd]] + args.cmd_args])
        return

    print(f"Unknown command: {cmd}")
    print("Run 'python run.py steps' for available commands.")
    sys.exit(1)


if __name__ == "__main__":
    main()
