# SEM 9 Repository Restructure Plan

## Goal

Turn this monorepo from a mixed coursework dump into a clear, navigable layout where:

1. **Hub + CTI + CV** stay the product surface and keep working.
2. **Spatial Lab** stays hub-excluded but follows one exercise layout.
3. **Theory / research / cognitive** coursework is grouped and no longer competes with lab paths at the root.
4. **Junk, duplicates, and nested environments** are removed or gitignored so disk and git stay honest.

This plan prioritizes **safe cleanup first**, then **hub path centralization**, then optional **renames**. It does not change product behavior unless a path move requires a coordinated code update.

---

## Current State (Summary)

### What works

| Area | Strength |
|---|---|
| Hub package (`hub/`) | Clean FastAPI layout; metadata-driven CTI/CV cards |
| CTI projects | Consistent `NN-kebab-case` IDs matching hub registry (18) |
| CV experiments | Clear `experiment_*` / `run_N` / `program_*` split; `_common.py` uses package-relative paths |
| Spatial exercises 01–07 | Mostly `project/` + `data/source\|processed` + `outputs/` |
| Spatial LFS | `.gitattributes` covers major binary types under `Spatial Lab/**` |

### What hurts

| Pain | Evidence |
|---|---|
| Kitchen-sink root | Labs, theory, research, cognitive, tmp, output, assets, empty junk dirs all siblings |
| Spaces in critical paths | `CTI LAB`, `CV LAB`, `Spatial Lab` — hub hardcodes these strings |
| Duplicate / stray CV trees | Root `assets/` (partial), root `outputs/` (empty shell), real trees live under `CV LAB/` |
| Nested venv bomb | `CTI LAB/03-phishing-url-detector/venv/` (~16k files); root `.venv` is the real env |
| Spatial legacy clutter | `Bengaluru_Flood_QGIS/`, `Coimbatore/`, `ex/`, `Exercise 4 Thematic Maps/` beside canonical exercises |
| Theory typo | `CV Thoery` vs `CTI Theory` |
| Doc drift | `AGENTS.md` says port **9000**; `run.py` serves **9999** |
| Absolute Windows paths | Spatial Ex03 tools; CTI 17 `users.json` embeddings |
| Mixed ignore policy | Research fully gitignored; Spatial carefully tracked; CTI data/results uneven |
| Dual deps | Root `requirements.txt` + many per-CTI `requirements.txt` |

### Hub path coupling (must not break)

| File | Hardcoded assumption |
|---|---|
| `hub/runner.py` | `SEM9/"CTI LAB"/id`, fallback `SEM9/"CV LAB"`, `.venv/Scripts/python.exe` |
| `hub/main.py` | mounts `CV LAB/outputs` as `/cv-outputs` |
| `hub/routers/api.py` | lists `CV LAB/outputs` |
| `hub/routers/cv.py` | 12× `launch_cmd` with `"CV LAB\\run_N.py"` |
| `run.py` | imports `hub.main:app`; hub must stay importable from root |

**Constraint:** Keep `hub/` one level under the monorepo root (or fix `SEM9 = parents[N]` if moved). Do not wire Spatial into the hub.

---

## Design Principles

1. **Root = product + lab entry points only** — hub, lab folders, shared config, docs that agents need.
2. **One home per concern** — no second `assets/`, `outputs/`, or theory folder at root.
3. **Rename only when payoff beats breakage** — spaces-in-names are annoying; coordinated renames touch hub, docs, LFS, and muscle memory.
4. **Move coursework as units** — theory/research/cognitive have no hub coupling; they can relocate freely.
5. **Spatial stays independent** — clean layout + LFS protocol; still excluded from hub routes.
6. **Single shared venv** — root `.venv` only; delete nested project venvs.
7. **Prefer relative paths** — scripts resolve from `__file__`, never machine-absolute paths.

---

## Target Layout

### Recommended end state (after Phases 0–2; renames optional in Phase 3)

```text
SEM 9/
├── AGENTS.md
├── PRODUCT.md
├── README.md                 # NEW: human root map (optional but useful)
├── RESTRUCTURE_PLAN.md       # this document
├── requirements.txt
├── run.py
├── hub/                      # FastAPI hub (unchanged location)
├── CTI LAB/                  # or cti-lab/ after Phase 4
├── CV LAB/                   # or cv-lab/ after Phase 4
├── Spatial Lab/              # or spatial-lab/ after Phase 4
│   └── Exercise NN - Name/
│       ├── project/
│       ├── data/source/
│       ├── data/processed/
│       ├── outputs/
│       ├── scripts/          # optional
│       ├── archive/          # optional legacy
│       └── README.md
├── coursework/               # NEW umbrella for non-hub work
│   ├── cti-theory/           # from CTI Theory/
│   ├── cv-theory/            # from CV Thoery/ (fixed spelling)
│   └── cognitive-computing/  # from Cognitive Computing/
├── research/                 # optional rename of long research dir (still gitignored if desired)
├── .venv/
├── .gitignore
└── .gitattributes
```

### Explicitly removed from root

| Remove / relocate | Why |
|---|---|
| `-p/`, `9/` | Empty junk (already gitignored) |
| Root `tmp/` | Scratch; already gitignored — keep local-only or delete |
| Root `assets/` | Partial duplicate of `CV LAB/assets/` |
| Root `outputs/` | Stray empty CV-style tree |
| Root `output/` | Mixed audit/PDF noise → `coursework/_scratch/` or delete |
| Loose `*.webm` | Media junk (already gitignored via `*.webm`) |
| Nested CTI `venv/` | Use root `.venv` |
| Spatial legacy dirs | Merge useful bits into exercise `archive/`, delete empties |

### What stays put (Phases 0–3)

- `hub/`, `run.py`, `CTI LAB/`, `CV LAB/` names and hub package depth
- Flask apps 03/06/16/18 ports and independence
- CV OpenCV desktop-only constraint
- Do-not-touch dirs: `.commandcode/`, `.hallmark/`, `.hermes/`, `.vscode/`, `.factory/`, `.git/`

---

## Phased Implementation

### Phase 0 — Inventory & safety rails (no moves)

**Purpose:** Make cleanup reversible and prevent accidental commits of junk.

| Step | Action |
|---|---|
| 0.1 | Snapshot: `git status --short`, note untracked trees (`Cognitive Computing/`, Spatial Ex 06–07, etc.) |
| 0.2 | Confirm nested venv is ignored: `git check-ignore -v "CTI LAB/03-phishing-url-detector/venv"` (already matches `venv/`) |
| 0.3 | Extend `.gitignore` for root clutter not yet covered: `output/`, `outputs/` (root only); keep `CV LAB/outputs/` tracked as today |
| 0.4 | Document dual-deps policy in root README or AGENTS: **root `requirements.txt` is source of truth**; per-project files are historical pins |
| 0.5 | Fix AGENTS port drift: 9000 → **9999** (docs only) |

**Exit criteria:** Ignore rules cover known junk; docs match `run.py`; no file moves yet.

**Risk:** None.

---

### Phase 1 — Low-risk cleanup (no hub code changes)

**Purpose:** Shrink noise without touching path strings the hub depends on.

#### 1A. Delete disposable root junk

- Delete empty `-p/`, `9/`
- Delete root loose webm if still present
- Delete or leave gitignored `tmp/` (local decision; do not commit)
- Delete root `outputs/` if empty / no unique artifacts
- Compare root `assets/` vs `CV LAB/assets/`:
  - Root `assets/datasets/defect_synth` has **40** files; CV LAB has **144**
  - Prefer CV LAB as canonical; delete root `assets/` after confirming no script writes to repo-root `assets/`
  - CV code already resolves assets via `CV LAB/_common.py` → package-relative `assets/`

#### 1B. Remove nested CTI venv

- Delete `CTI LAB/03-phishing-url-detector/venv/` from disk
- Verify project runs under root `.venv`
- Ensure `venv/` remains in `.gitignore` (already)

#### 1C. CTI internal hygiene (per project, non-breaking)

| Clean | Example |
|---|---|
| Smoke/debug artifacts | `02/_smoke.py`, `_smoke.out`, `_train.out` |
| Logs | `03/server.log` (already `*.log` ignored) |
| Absolute paths | Regenerate `17/.../data/users.json` with relative embedding paths |

Do **not** bulk-delete `data/` or `results/` without checking which artifacts are needed for demos.

#### 1D. Spatial Lab legacy consolidation

| Path | Action |
|---|---|
| `Spatial Lab/ex/` | Delete if empty |
| `Spatial Lab/Exercise 4 Thematic Maps/` | Delete if empty duplicate of Ex 04 |
| `Spatial Lab/Bengaluru_Flood_QGIS/` | If no unique layers vs Ex 03, delete or move leftovers into `Exercise 03/.../archive/` |
| `Spatial Lab/Coimbatore/` (`Best/`, `New folder/`) | Inspect once; archive unique files into relevant Exercise 01/06/07 `archive/`, then delete informal dump |
| Ex 03 `tools/*.py` | Replace absolute `DATA_DIR = r"C:\Users\..."` with paths relative to exercise root |

Normalize layout gaps:

- Ex 02: optionally group gpkg under `data/processed/` (or document why flat `data/` is fine)
- Keep Ex 01–07 README index in `Spatial Lab/README.md` accurate

#### 1E. Theory rename (hub-safe)

- Rename `CV Thoery` → `CV Theory` (or wait for Phase 2 move into `coursework/cv-theory/`)

**Exit criteria:** Root shows only meaningful top-level dirs; Spatial has only Exercise 01–07 (+ README); no nested venv; hub still runs unchanged.

**Risk:** Low. Spatial archive decisions need a quick human glance before delete.

**Verification:**

```powershell
.venv\Scripts\python.exe run.py
# smoke hub routes
curl -s http://127.0.0.1:9999/ | Select-String "Lab Hub"
curl -s http://127.0.0.1:9999/cti | Select-String "projects"
curl -s http://127.0.0.1:9999/cv | Select-String "experiments"
```

---

### Phase 2 — Group non-lab coursework under `coursework/`

**Purpose:** Make root about the hub + three labs.

| From | To |
|---|---|
| `CTI Theory/` | `coursework/cti-theory/` |
| `CV Thoery/` or `CV Theory/` | `coursework/cv-theory/` |
| `Cognitive Computing/` | `coursework/cognitive-computing/` |
| Optional: root `output/` keepers | `coursework/_artifacts/` or delete |

Research tree options:

| Option | When |
|---|---|
| **A.** Leave as-is (long name, fully gitignored) | Least churn |
| **B.** Move to `research/healthcare-multi-agent/` and keep gitignored | Cleaner root |
| **C.** Split into its own repo later | If research grows further |

**Hub impact:** None — hub does not reference these paths.

**Docs impact:** Update any local notes that point at old theory paths; AGENTS Spatial/CTI/CV sections unchanged.

**Exit criteria:** Root top-level = hub product + labs + coursework(+research) + config.

**Risk:** Low–medium only if external notes/bookmarks use old paths; no code coupling found.

---

### Phase 3 — Hub path centralization (prep for renames)

**Purpose:** Make future renames a one-place change instead of grepping 12 launch commands.

#### 3.1 Add path constants module

Create something like `hub/paths.py`:

```python
from pathlib import Path

SEM9 = Path(__file__).resolve().parent.parent
VENV_PYTHON = SEM9 / ".venv" / "Scripts" / "python.exe"
CTI_LAB = SEM9 / "CTI LAB"
CV_LAB = SEM9 / "CV LAB"
CV_OUTPUTS = CV_LAB / "outputs"
```

Wire into:

- `hub/runner.py` (`_resolve_script`, `VENV_PYTHON`)
- `hub/main.py` (CV outputs mount)
- `hub/routers/api.py`
- `hub/routers/cv.py` — build `launch_cmd` from `CV_LAB` + `run_N.py` instead of duplicated string literals

#### 3.2 Optional: single public CV entry style

Pick one documented interface for humans + hub:

| Choice | Pros |
|---|---|
| **Keep `run_N.py` as public API** (recommended short-term) | Already in hub metadata; wrappers are thin |
| Hub calls `experiment_*.py` directly | Fewer files long-term; larger metadata churn |

Do not delete `experiment_*.py` either way — packages and imports depend on them.

#### 3.3 Docs alignment

- `AGENTS.md` architecture tree matches reality
- Curl examples use port **9999** and current script args (`run.py` + `args=all` where applicable)
- Root `README.md` maps folders in one screen

**Exit criteria:** All hub path joins go through `hub/paths.py`; hub smoke tests pass.

**Risk:** Medium (code change, no folder rename yet). Easy rollback.

---

### Phase 4 — Optional renames (high coordination)

Only do this if spaces/quoting pain is worth a dedicated pass.

| Current | Proposed | Touches |
|---|---|---|
| `CTI LAB` | `cti-lab` | `hub/paths.py`, CTI README, AGENTS, any docs, chrome-extension copy if any |
| `CV LAB` | `cv-lab` | `hub/paths.py`, all launch_cmds (auto if centralized), CV README, `.gitattributes` CV lines, `.gitignore` model paths |
| `Spatial Lab` | `spatial-lab` | `.gitattributes` `Spatial Lab/**`, `.gitignore` Spatial rules, Spatial README, QGIS relative paths (usually OK if relative), tools |

**Procedure for each rename:**

1. Update `hub/paths.py` (and any remaining string refs) in the same commit as `git mv`
2. Use `git mv` to preserve history
3. For Spatial: update `.gitattributes` / `.gitignore` path prefixes; run `git lfs ls-files`, `git lfs fsck`
4. Reopen a sample QGIS project and confirm relative data sources
5. Run hub smoke + one CTI `api/run` + one CV launch_cmd string check

**Do not** rename individual CTI project folders unless hub `PROJECTS[].id` and URLs are updated together (`/cti/{id}`).

**Risk:** High. Recommend a single focused PR/commit, not mixed with feature work.

---

### Phase 5 — Data & dependency policy (ongoing hygiene)

| Policy | Rule |
|---|---|
| Shared env | Only root `.venv`; never commit project-local venvs |
| CTI data | Large downloadable corpora (e.g. SpamAssassin) prefer download scripts + gitignore raw dumps when possible |
| CTI results | Commit only demo-sized metrics/models needed for “works offline”; ignore bulky regenerable artifacts if size hurts |
| CV models | Keep large weights gitignored or LFS as today; document download in `CV LAB/assets/**/README` |
| Spatial | Continue LFS + explicit path staging; no hub integration |
| Coursework | Prefer LaTeX sources tracked; build aux + giant PNG page dumps stay ignored under `tmp/` |

---

## What We Will Not Do

| Out of scope | Why |
|---|---|
| Wire Spatial into hub routes | AGENTS hard constraint |
| Convert Flask apps 03/06/16/18 to FastAPI | Independent legacy UIs |
| Embed OpenCV GUI into browser | CV requires desktop display |
| Edit do-not-touch agent/config dirs | Tool state / git metadata |
| `git add -A` for Spatial | Risk of committing unrelated coursework |
| Force-push / history rewrite for renames | Routine structure work must stay additive |
| Flatten all CTI projects into one package | Each project is a graded unit with its own `run.py` |

---

## Suggested Priority Order

| Priority | Phase | Effort | Hub break risk | Payoff |
|---|---|---|---|---|
| P0 | Phase 0 docs + ignore | S | None | Correct ops docs |
| P1 | Phase 1 cleanup | S–M | None | Disk + clarity |
| P2 | Phase 2 `coursework/` | S | None | Clean root mental model |
| P3 | Phase 3 path constants | S | Low | Enables safe renames later |
| P4 | Phase 4 renames | M | High if sloppy | Shell ergonomics |
| P5 | Phase 5 policies | Ongoing | Low | Long-term size control |

**Recommended default execution:** Phases **0 → 1 → 2 → 3**. Defer Phase 4 unless you explicitly want kebab-case lab folder names.

---

## Implementation Commit Slice Plan

Prefer small commits:

1. **docs-ignore** — AGENTS port fix, `.gitignore` root `output`/`outputs`, optional root README
2. **cleanup-root** — delete junk, root assets/outputs, nested venv
3. **cleanup-spatial** — legacy dirs + relative tools paths
4. **coursework-move** — theory + cognitive under `coursework/`
5. **hub-paths** — `hub/paths.py` + wire-up (no renames)
6. **rename-labs** (optional) — `git mv` + path constant updates + LFS attrs

---

## Verification Checklist (after any structural change)

```powershell
# Hub
.venv\Scripts\python.exe run.py
curl -s http://127.0.0.1:9999/
curl -s http://127.0.0.1:9999/cti
curl -s http://127.0.0.1:9999/cv
curl -s -m 15 "http://127.0.0.1:9999/api/run/01-spam-ham-watcher?script=run.py&args=--help"

# Path resolution still finds CTI then CV
# (after Phase 3, constants point at real dirs)

# Spatial (if touched)
git lfs install --local
git check-attr filter diff merge -- "Spatial Lab/Exercise 05 - Nilgiris Check-Dam Suitability/project/exercise-05-nilgiris-check-dam.qgz"
git lfs ls-files
git status --short
```

Manual: open one QGIS exercise project and confirm layers resolve relatively.

---

## Open Decisions (at execution time)

1. **Phase 4 renames?** Default **no** — keep `CTI LAB` / `CV LAB` / `Spatial Lab` names after cleanup + path constants.
2. **Research folder?** Default leave gitignored long name, or move under `research/` for root clarity.
3. **CTI `results/` commit policy?** Keep demo artifacts vs ignore regenerable models — decide per project size pain.
4. **Root `tmp/` / `output/`?** Delete locally vs keep as personal scratch (ignored).

---

## Success Criteria

- Root directory lists a short, intentional set of names (hub product + labs + coursework).
- No nested venvs; no duplicate root `assets/`/`outputs/`.
- Spatial contains only numbered exercises (+ README), with legacy under `archive/` where needed.
- Hub still resolves all 18 CTI + 12 CV entries; SSE run still works.
- AGENTS/README paths and port match reality.
- Future lab renames, if ever done, require editing primarily `hub/paths.py` (and LFS attrs for Spatial).
