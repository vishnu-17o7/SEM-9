"""Entry point for the SEM 9 Lab Hub.

Usage:
    python run.py
    # Opens http://127.0.0.1:9999
"""
from __future__ import annotations

import os
from pathlib import Path

import uvicorn
from dotenv import load_dotenv

# Load .env from the project root
ENV_PATH = Path(__file__).resolve().parent / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)

# Warn if Supermemory is not configured
if not os.environ.get("SUPERMEMORY_API_KEY"):
    print("[info] SUPERMEMORY_API_KEY not set -- memory persistence disabled")

if __name__ == "__main__":
    uvicorn.run("hub.main:app", host="127.0.0.1", port=9999, reload=False)
