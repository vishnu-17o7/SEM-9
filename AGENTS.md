# SEM 9 Lab Hub

Unified FastAPI dashboard for the CTI Lab (18 cyber threat intelligence projects) and CV Lab (12 computer vision experiments). Each project or experiment has a detail page with description, controls, and (for CTI) Run buttons that execute scripts via subprocess with SSE streaming output.

## Core Commands

All commands run from the project root using the virtual environment interpreter.

- **Start the hub**: `.venv\Scripts\python.exe run.py` then open `http://127.0.0.1:9000`
- **Install a new package**: `.venv\Scripts\python.exe -m pip install <pkg>`
- **Freeze dependencies**: `.venv\Scripts\python.exe -m pip freeze > requirements.txt`
- **Run any CTI/CV script directly**: `.venv\Scripts\python.exe "CTI LAB\<project>\<script>.py"` or `.venv\Scripts\python.exe "CV LAB\experiment_<n>_<name>.py"`

## Architecture Overview

```
SEM 9/
├── hub/                          # FastAPI hub package
│   ├── main.py                   # FastAPI app, mounts routers + static files
│   ├── templates.py              # Raw Jinja2 Environment (bypasses starlette cache bug)
│   ├── runner.py                 # Thread-based subprocess runner → SSE streaming
│   ├── static/
│   │   ├── style.css             # Design system tokens (teal, from CV tokens.css)
│   │   └── script.js             # SSE client + Run button handler
│   ├── templates/
│   │   ├── base.html             # Sidebar nav (CTI | CV), main content slot
│   │   ├── index.html            # Landing page with category cards
│   │   ├── cti/index.html        # 18-project card grid
│   │   ├── cti/project.html      # Detail page + Run buttons per script
│   │   ├── cv/index.html         # 12-experiment card grid
│   │   └── cv/experiment.html    # Detail page + launch command
│   └── routers/
│       ├── api.py                # GET /api/run/{id}?script=...&args=... → SSE stream
│       ├── cti.py                # GET /cti (index), /cti/{id} (detail)
│       └── cv.py                 # GET /cv (index), /cv/{id} (detail)
├── CTI LAB/                      # 18 project subdirectories (01-18)
├── CV LAB/                       # 12 experiment_*.py files + program_* packages
└── run.py                        # uvicorn entry point
```

**Data flow for Run button**: Browser -> `GET /api/run/{project_id}?script=<path>` -> `runner.run_script()` spawns `subprocess.Popen` in a daemon thread -> lines pushed to `asyncio.Queue` -> SSE stream -> browser appends lines to output div.

## Routes

| Path | Description |
|---|---|
| `GET /` | Landing page with CTI / CV category cards |
| `GET /cti` | Grid of all 18 CTI projects |
| `GET /cti/{id}` | CTI project detail page with Run buttons |
| `GET /cv` | Grid of all 12 CV experiments |
| `GET /cv/{id}` | CV experiment detail page with launch command |
| `GET /api/run/{id}?script=...&args=...` | SSE streaming endpoint for script execution |

## Conventions & Patterns

- **Templates**: rendered via `hub/templates.py`'s `render(name, request=..., ...)` using a raw `jinja2.Environment` (not starlette's `Jinja2Templates`, which has a cache-key bug in Jinja2 3.1.x).
- **Static files**: served by FastAPI at `/static/`. All asset paths are absolute from `/static/`.
- **Project metadata**: defined as `dict` lists in `hub/routers/cti.py` (`PROJECTS`) and `hub/routers/cv.py` (`EXPERIMENTS`). Adding an entry is sufficient to render a card + detail page — no code changes beyond the metadata dict.
- **Script resolution**: `runner._resolve_script()` checks `CTI LAB/{project_id}/` first, then `CV LAB/` as fallback. The `project_id` in the router metadata must match the directory name under `CTI LAB/` exactly.
- **SSE format**: each chunk is JSON: `{"text": "...", "stream": "stdout"|"stderr"}`. End-of-stream is signaled by an `event: done` SSE event.
- **Windows subprocess**: `runner.py` uses `subprocess.Popen` + daemon threads + `asyncio.Queue` to avoid `ProactorEventLoop` limitations.
- **Entry point**: `run.py` starts uvicorn with `reload=False` (reload breaks async subprocess on Windows; restart manually after edits).

## How to Add a New Project

### CTI project

1. Create the project directory under `CTI LAB/<id>/`
2. Open `hub/routers/cti.py` and append to `PROJECTS`:
   ```python
   {
       "id": "19-my-project",
       "num": "19",
       "title": "My Project",
       "type": "cli",
       "port": None,
       "description": "What this project does.",
       "scripts": [
           {"path": "script.py", "desc": "What this script does"},
       ],
   }
   ```
3. For scripts needing CLI args, add `"args": "--flag value"` to the script entry.
4. For web UI projects, set `"type": "web"` and `"port": <int>` instead.

### CV experiment

1. Create `CV LAB/experiment_<n>_<name>.py` (or a `program_<n>_<name>/` package)
2. Open `hub/routers/cv.py` and append to `EXPERIMENTS`:
   ```python
   {
       "id": "experiment_13_name",
       "num": "13",
       "title": "My Experiment",
       "description": "What this experiment demonstrates.",
       "launch_cmd": '.venv\\Scripts\\python.exe "CV LAB\\experiment_13_name.py"',
       "has_package": True,
       "extra_args": [],
       "ui_controls": "Keyboard controls description.",
   }
   ```

## Requirements.txt

Located at root `SEM 9/requirements.txt`. Update after installing any new package:

```
.venv\Scripts\python.exe -m pip install <new-pkg>
.venv\Scripts\python.exe -m pip freeze > requirements.txt
```

## Tech Stack (Stick to FastAPI)

The hub is built on **FastAPI** — not Flask, not Django. All routes, API endpoints, and new components must use FastAPI patterns:

- `APIRouter` with `prefix` and `tags`
- Type-annotated path and query parameters on route handlers
- `HTMLResponse` for page routes, `StreamingResponse` for SSE endpoints
- `@router.get()`, `@router.post()` decorators with `response_class=`

Do not introduce Flask (`Flask`, `render_template`, `render_template_string`), Django, or any other web framework into the hub package. The four legacy Flask apps (CTI 03, 06, 16, 18) run independently on their own ports and are not part of the hub's codebase.

## Testing & Verification

After making changes to the hub, verify all routes respond correctly:

```
# Start the server
.venv\Scripts\python.exe run.py

# In another terminal, test each route
curl -s http://127.0.0.1:9000/ | Select-String "SEM 9 Lab Hub"
curl -s http://127.0.0.1:9000/cti | Select-String "18 projects"
curl -s http://127.0.0.1:9000/cti/01-spam-ham-watcher | Select-String "Spam/Ham Watcher"
curl -s http://127.0.0.1:9000/cv | Select-String "12 experiments"
curl -s http://127.0.0.1:9000/cv/experiment_6_yolo_maskrcnn | Select-String "YOLOv3"
curl -s http://127.0.0.1:9000/static/style.css | Select-String "color-accent"

# Test the Run API with a quick script
curl -s -m 10 "http://127.0.0.1:9000/api/run/01-spam-ham-watcher?script=classify.py"
# Expected: SSE events with "Usage:  python classify.py <text>" and "event: done"

# Test Run API with arguments
curl -s -m 10 "http://127.0.0.1:9000/api/run/14-threat-intel-repository?script=cli_query.py&args=--stats"
```

## CV Experiments Constraint

All 12 CV experiments use native OpenCV GUI windows (`cv2.imshow`, `cv2.waitKey`, etc.). They **require a physical display** and an interactive desktop environment — they cannot run headless, in a container without display, or inside the web browser. The hub only shows experiment descriptions and launch commands. Do not attempt to embed, stream, or proxy OpenCV output into the hub's web interface.

## Legacy Flask Apps (CTI 03, 06, 16, 18)

Four CTI projects have their own Flask web servers that run independently on separate ports:

| Project | File | Port |
|---|---|---|
| 03 Phishing URL Detector | `CTI LAB/03-phishing-url-detector/app.py` | 5000 |
| 06 UEBA Dashboard | `CTI LAB/06-behavioral-profile-ueba/dashboard.py` | 5001 |
| 16 Threat Intel Dashboard | `CTI LAB/16-threat-intel-dashboard/app.py` | 5002 |
| 18 Multi-Factor Auth | `CTI LAB/18-multi-factor-biometric-auth/app.py` | 5003 |

The hub links to them via an "Open Web UI" button on their detail pages but does not own, rewrite, or migrate them. Do not convert these Flask apps to FastAPI unless explicitly asked.

## Spatial Lab Exclusion

`SEM 9/Spatial Lab/` contains QGIS GeoPackage files (`*.gpkg`, `*.qgz`) and is **not part of the hub**. Do not create routes, pages, runner entries, or any integration for Spatial Lab projects. The hub covers only CTI LAB and CV LAB.

## Spatial Lab Git and Git LFS Protocol

Spatial Lab is versioned in this repository, but it remains independent of the FastAPI hub.

- Keep each exercise under `Spatial Lab/Exercise NN - Name/` with `project/`, `data/source/`, `data/processed/`, and `outputs/` as needed.
- Save QGIS projects inside the exercise `project/` directory and use relative data-source paths. Never commit a project that points to temporary processing folders, Downloads, attachments, or another machine's absolute paths.
- Store large or binary spatial artifacts through Git LFS. The repository `.gitattributes` covers Spatial Lab `gpkg`, `tif/tiff`, `mbtiles`, `geojson`, `kml`, `osm`, `osm.gz`, `qgz`, `pdf`, `png`, and `jpg` files.
- Before staging Spatial Lab work, run `git lfs install --local` and confirm relevant files with `git check-attr filter diff merge -- "Spatial Lab/<path>"`.
- Do not commit live QGIS lock or journal files such as `.qgis/`, `*.gpkg-shm`, `*.gpkg-wal`, or `*.qgz~`.
- Stage explicit Spatial Lab paths instead of `git add -A`, so unrelated coursework and local files are not included accidentally.
- Before committing, verify the project reopens with valid file-backed layers, then run `git lfs ls-files`, `git lfs fsck`, `git diff --cached --check`, and `git status --short`.
- Before pushing, fetch `origin`, confirm the intended branch is up to date, and push the verified commit. Never rewrite shared history for routine Spatial Lab updates.

## Do Not Touch Hidden Config Dirs

The following directories contain automated tool state and git metadata. Never create, edit, or delete files inside them:

- `.commandcode/` at project root and in `CTI LAB/`
- `.hallmark/` in `CV LAB/`
- `.hermes/` in `CTI LAB/`
- `.vscode/` in `CV LAB/`
- `.factory/` at user home or project root
- `.git/` at project root

## Code Style

- **`from __future__ import annotations`** at the top of every Python file
- **Import order**: stdlib -> third-party -> local modules; single blank line between groups; one `import` per line
- **Strings**: use f-strings, never `%` formatting or `.format()` calls
- **Type annotations**: annotate all function signatures (PEP 484). Use `list[str]` not `List[str]`, `dict` not `Dict`, etc.
- **Docstrings**: triple-quoted `"""..."""` on the first line inside the function body, no blank line after the opening `"""`
- **Routes**: `async def` handlers with type-annotated parameters and explicit `response_class=HTMLResponse` on the decorator
- **Immutability**: do not modify `PROJECTS` or `EXPERIMENTS` lists in-place during request handling (they are module-level constants after startup)
