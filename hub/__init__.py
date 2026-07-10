"""SEM 9 Lab Hub package."""
import sys

# Windows ASGI servers (uvicorn reload) need ProactorEventLoop for subprocess
if sys.platform == "win32":
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())