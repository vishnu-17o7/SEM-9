"""
Custom template rendering using raw Jinja2 — bypasses starlette's Jinja2Templates
which has a cache-key-hashing bug in Jinja2 3.1+.
"""
from pathlib import Path

import jinja2

TEMPLATES_DIR = Path(__file__).parent / "templates"
_env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(str(TEMPLATES_DIR)),
    autoescape=True,
)


def render(name: str, **context: object) -> str:
    """Render a template and return the HTML string."""
    template = _env.get_template(name)
    return template.render(**context)

