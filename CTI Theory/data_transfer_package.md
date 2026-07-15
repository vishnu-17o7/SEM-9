# CTI Theory Report Reproducibility Package

This directory contains the auditable source package for the CTI adversary-profile report.

## Canonical files

- `distillation_attacks_adversary_profile.tex` - canonical report source
- `distillation_attacks_adversary_profile.pdf` - compiled submission
- `distillation_attack_navigator.json` - confidence-rated ATT&CK Navigator layer
- `evidence_register.csv` - claim-level source and limitation register
- `generate_report_artifacts.py` - deterministic figure generator

`distillation_attacks_adversary_profile.md` is a short compatibility note. It is not the canonical report source.

## Regenerate figures

From the project root:

```powershell
.venv\Scripts\python.exe "CTI Theory\generate_report_artifacts.py"
```

This regenerates:

- `distillation_attack_scale.png`
- `attack_technique_distribution.png`
- `distillation_attack_navigator.png`

The exchange totals are declared in the generator and trace to claim IDs C01-C07 in `evidence_register.csv`. The Navigator summary is generated from `distillation_attack_navigator.json`.

## Compile report

```powershell
pdflatex -interaction=nonstopmode -halt-on-error -output-directory="CTI Theory" "CTI Theory\distillation_attacks_adversary_profile.tex"
pdflatex -interaction=nonstopmode -halt-on-error -output-directory="CTI Theory" "CTI Theory\distillation_attacks_adversary_profile.tex"
```

## Interpretation boundary

- Campaign scale and attribution are provider-reported unless stated otherwise.
- Repetition of a provider figure by news outlets is corroboration of the disclosure, not independent measurement.
- ATT&CK entries are analyst-created analogies with confidence caveats.
- MITRE ATLAS is used for AI-native model access and extraction behavior.
- Public sources do not expose the raw telemetry needed to reproduce Anthropic's attribution.
