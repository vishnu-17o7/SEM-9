"""
Subprocess runner that uses a thread to run a process and streams
output lines back to the async SSE handler via an asyncio.Queue.
"""
import asyncio
import subprocess
import threading
from pathlib import Path
from typing import AsyncGenerator


SEM9 = Path(__file__).resolve().parent.parent
VENV_PYTHON = SEM9 / ".venv" / "Scripts" / "python.exe"


async def run_script(
    project_id: str,
    script_path: str,
    args: list[str] | None = None,
) -> AsyncGenerator[dict, None]:
    """Run a Python script in a thread and yield SSE events."""
    resolved = _resolve_script(project_id, script_path)
    if not resolved:
        yield {"text": f"Script not found: {script_path}\n", "stream": "stderr"}
        return

    cmd = [str(VENV_PYTHON), str(resolved)]
    if args:
        cmd.extend(args)

    queue: asyncio.Queue[dict | None] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def _run_and_enqueue():
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(resolved.parent),
            text=True,
            bufsize=1,
        )

        def _read(stream, name):
            for line in iter(stream.readline, ""):
                asyncio.run_coroutine_threadsafe(
                    queue.put({"text": line, "stream": name}),
                    loop,
                )
            stream.close()

        out_thread = threading.Thread(target=_read, args=(proc.stdout, "stdout"), daemon=True)
        err_thread = threading.Thread(target=_read, args=(proc.stderr, "stderr"), daemon=True)
        out_thread.start()
        err_thread.start()

        proc.wait()
        out_thread.join()
        err_thread.join()

        if proc.returncode != 0:
            asyncio.run_coroutine_threadsafe(
                queue.put({
                    "text": f"\n[process exited with code {proc.returncode}]\n",
                    "stream": "stderr",
                }),
                loop,
            )

        asyncio.run_coroutine_threadsafe(queue.put(None), loop)

    thread = threading.Thread(target=_run_and_enqueue, daemon=True)
    thread.start()

    while True:
        chunk = await queue.get()
        if chunk is None:
            break
        yield chunk

    thread.join()


def _resolve_script(project_id: str, script_path: str) -> Path | None:
    """Resolve a script path relative to its project directory."""
    base = SEM9 / "CTI LAB" / project_id
    if base.exists():
        candidate = base / script_path
        if candidate.exists():
            return candidate
        if not script_path.endswith(".py"):
            candidate = base / f"{script_path}.py"
            if candidate.exists():
                return candidate
    base_cv = SEM9 / "CV LAB"
    if base_cv.exists():
        candidate = base_cv / script_path
        if candidate.exists():
            return candidate
        if not script_path.endswith(".py"):
            candidate = base_cv / f"{script_path}.py"
            if candidate.exists():
                return candidate
    return None
