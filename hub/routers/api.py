"""Shared API — script runner SSE endpoint and CV output listing."""
from __future__ import annotations

import json
import re
from pathlib import Path

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse, StreamingResponse

from hub.runner import run_script
from hub.supermemory import memorize

router = APIRouter(prefix="/api", tags=["api"])

# Path to CV LAB outputs
SEM9 = Path(__file__).resolve().parent.parent.parent
CV_OUTPUTS = SEM9 / "CV LAB" / "outputs"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".webm"}
DATA_EXTS = {".json", ".csv", ".npz", ".pth", ".joblib"}


@router.get("/run/{project_id}")
async def api_run(
    project_id: str,
    script: str = Query(...),
    args: str = Query(default=""),
) -> StreamingResponse:
    """SSE stream of script output."""

    # Look up project metadata for richer memorisation
    from hub.routers.cti import CTI_PROJECT_MAP
    from hub.routers.cv import CV_EXPERIMENT_MAP

    meta: dict[str, str] | None = CTI_PROJECT_MAP.get(project_id) or CV_EXPERIMENT_MAP.get(project_id)
    project_title = meta.get("title", project_id) if meta else project_id
    script_desc = next(
        (s["desc"] for s in (meta.get("scripts", []) if meta else []) if s.get("path") == script),
        script,
    )

    arg_list = args.split() if args else None

    async def event_stream():
        exit_code = 0

        async for chunk in run_script(project_id, script, arg_list):
            yield f"data: {json.dumps(chunk)}\n\n"
            if chunk.get("stream") == "stderr" and chunk.get("text", "").startswith("[process exited with code "):
                m = re.search(r"code (\d+)", chunk["text"])
                if m:
                    exit_code = int(m.group(1))

        # Memorise the run outcome before sending done, so it persists
        # even if the client disconnects immediately after the final event.
        content = (
            f"Ran script '{script_desc}' ({script}) for project '{project_title}' "
            f"(id: {project_id})"
        )
        if args:
            content += f" with args: {args}"
        content += f". Exit code: {exit_code}"
        content += " (success)" if exit_code == 0 else " (failure)"

        memorize(
            content=content,
            container_tag="sem9-hub",
            metadata={
                "project_id": project_id,
                "project_title": project_title,
                "script": script,
                "script_desc": script_desc,
                "args": args,
                "exit_code": exit_code,
                "type": "script_run",
            },
        )

        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/outputs/{experiment_id}", response_class=JSONResponse)
async def list_outputs(experiment_id: str) -> JSONResponse:
    """List saved output files for a CV experiment."""
    out_dir = CV_OUTPUTS / experiment_id
    if not out_dir.exists():
        return JSONResponse({"error": "No outputs found"}, status_code=404)

    all_exts = IMAGE_EXTS | VIDEO_EXTS | DATA_EXTS
    files: list[dict[str, str | int]] = []

    # Recursive glob to pick up nested subdirectories too
    for f in sorted(out_dir.rglob("*")):
        if not f.is_file() or f.suffix.lower() not in all_exts:
            continue
        rel = f.relative_to(out_dir).as_posix()
        ext = f.suffix[1:].lower()

        if ext in ("png", "jpg", "jpeg", "webp"):
            category = "image"
        elif ext in ("gif",):
            category = "image"
        elif ext in ("mp4", "avi", "webm"):
            category = "video"
        else:
            category = "data"

        files.append({
            "name": f.name,
            "rel": rel,
            "path": f"/cv-outputs/{experiment_id}/{rel}",
            "size": f.stat().st_size,
            "type": ext,
            "category": category,
        })

    # Sort: images first, then videos, then data files
    order = {"image": 0, "video": 1, "data": 2}
    files.sort(key=lambda item: order.get(item["category"], 3))

    return JSONResponse(files)
