"""SEM 9 Hub — FastAPI application."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from hub.templates import render
from hub.routers.api import router as api_router
from hub.routers.cti import router as cti_router
from hub.routers.cv import router as cv_router
from hub.supermemory import recall

app = FastAPI(title="SEM 9 Lab Hub")

# Static files
HUB_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=HUB_DIR / "static"), name="static")

# CV Lab output images
SEM9_DIR = HUB_DIR.parent
CV_OUTPUTS_DIR = SEM9_DIR / "CV LAB" / "outputs"
if CV_OUTPUTS_DIR.exists():
    app.mount("/cv-outputs", StaticFiles(directory=str(CV_OUTPUTS_DIR)), name="cv-outputs")

# Routers
app.include_router(api_router)
app.include_router(cti_router)
app.include_router(cv_router)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    memories = recall("SEM 9 Lab Hub recent activity", limit=5)
    html = render("index.html", request=request, active="home", memories=memories)
    return HTMLResponse(html)
