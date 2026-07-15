"""Regenerate the CTI Theory report figures from declared source data."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_DIR = Path(__file__).parent
NAVIGATOR_PATH = PROJECT_DIR / "distillation_attack_navigator.json"
ATTACK_SCALE = {
    "DeepSeek": 150_000,
    "Moonshot AI": 3_400_000,
    "MiniMax": 13_000_000,
    "Alibaba/Qwen": 28_800_000,
}
TACTIC_LABELS = {
    "resource-development": "Resource Development",
    "persistence": "Persistence",
    "initial-access": "Initial Access",
    "command-and-control": "Command and Control",
    "exfiltration": "Exfiltration",
}
COLORS = {
    "navy": "#1B3A5C",
    "red": "#C0392B",
    "green": "#2A9D8F",
    "orange": "#F4A261",
    "coral": "#E76F51",
    "grid": "#D9E2EC",
}


def load_techniques() -> list[dict[str, object]]:
    """Load enabled techniques from the Navigator layer."""
    payload = json.loads(NAVIGATOR_PATH.read_text(encoding="utf-8"))
    return [technique for technique in payload["techniques"] if technique.get("enabled", True)]


def style_axis(axis: plt.Axes) -> None:
    """Apply the report's restrained visual style."""
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.grid(axis="x", color=COLORS["grid"], linewidth=0.8)
    axis.set_axisbelow(True)


def generate_attack_scale() -> None:
    """Render provider-reported exchange totals."""
    names = list(ATTACK_SCALE)
    values = [ATTACK_SCALE[name] / 1_000_000 for name in names]
    figure, axis = plt.subplots(figsize=(10, 5.4))
    bars = axis.barh(names, values, color=[COLORS["navy"], COLORS["navy"], COLORS["red"], COLORS["orange"]])
    axis.set_xlabel("Reported API exchanges (millions)")
    axis.set_title("Provider-Reported Distillation Campaign Scale", loc="left", weight="bold", pad=28)
    axis.text(0, 1.015, "Counts are allegations reported by Anthropic; raw telemetry is not public.", transform=axis.transAxes, color="#5B6770")
    axis.bar_label(bars, labels=[f"{value:,.2f}M" for value in values], padding=5)
    style_axis(axis)
    figure.tight_layout()
    figure.savefig(PROJECT_DIR / "distillation_attack_scale.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def generate_distribution(techniques: list[dict[str, object]]) -> None:
    """Render the reduced ATT&CK analogy distribution."""
    counts = Counter(str(technique["tactic"]) for technique in techniques)
    tactics = list(counts)
    values = [counts[tactic] for tactic in tactics]
    labels = [TACTIC_LABELS.get(tactic, tactic.replace("-", " ").title()) for tactic in tactics]
    figure, axis = plt.subplots(figsize=(10, 5.4))
    bars = axis.barh(labels, values, color=COLORS["navy"])
    axis.set_xlabel("Retained ATT&CK analogies")
    axis.set_title("Constrained ATT&CK Crosswalk by Tactic", loc="left", weight="bold", pad=28)
    axis.text(0, 1.015, "Five confidence-rated analogies; AI-native behavior is mapped to MITRE ATLAS.", transform=axis.transAxes, color="#5B6770")
    axis.set_xticks(range(0, max(values) + 1))
    axis.bar_label(bars, padding=5)
    style_axis(axis)
    figure.tight_layout()
    figure.savefig(PROJECT_DIR / "attack_technique_distribution.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def generate_navigator_summary(techniques: list[dict[str, object]]) -> None:
    """Render a readable companion summary for the Navigator JSON layer."""
    figure, axis = plt.subplots(figsize=(12, 6.6))
    axis.axis("off")
    axis.set_title("ATT&CK Navigator Layer — Confidence-Rated Analogies", loc="left", fontsize=17, weight="bold", color=COLORS["navy"], pad=18)
    axis.text(0, 0.96, "Analyst-created crosswalk; not an official MITRE or Anthropic campaign mapping.", transform=axis.transAxes, color="#5B6770", fontsize=10)
    row_height = 0.145
    for index, technique in enumerate(techniques):
        top = 0.85 - index * row_height
        score = int(technique["score"])
        confidence = "High" if score >= 75 else "Medium" if score >= 50 else "Low"
        tactic = TACTIC_LABELS.get(str(technique["tactic"]), str(technique["tactic"]))
        axis.add_patch(plt.Rectangle((0.0, top - 0.085), 0.98, 0.105, color="#F5F7FA", ec=COLORS["grid"], transform=axis.transAxes))
        axis.add_patch(plt.Rectangle((0.0, top - 0.085), 0.018, 0.105, color=str(technique["color"]), transform=axis.transAxes))
        axis.text(0.035, top - 0.025, str(technique["techniqueID"]), transform=axis.transAxes, weight="bold", color=COLORS["navy"], va="center")
        axis.text(0.18, top - 0.025, tactic, transform=axis.transAxes, va="center")
        axis.text(0.57, top - 0.025, f"{confidence} confidence analogy", transform=axis.transAxes, va="center", color=str(technique["color"]), weight="bold")
    figure.tight_layout()
    figure.savefig(PROJECT_DIR / "distillation_attack_navigator.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """Regenerate all report figures."""
    techniques = load_techniques()
    generate_attack_scale()
    generate_distribution(techniques)
    generate_navigator_summary(techniques)
    print(f"Regenerated 3 report figures from {NAVIGATOR_PATH.name}.")


if __name__ == "__main__":
    main()
