"""Supermemory integration -- persistent memory for the SEM 9 Lab Hub.

Provides ``memorize()`` to persist notable events and ``recall()`` to retrieve
relevant context.  All memories are scoped to the ``sem9-hub`` container tag
unless another is specified.
"""
from __future__ import annotations

import os

from supermemory import Supermemory

_CLIENT: Supermemory | None = None
_DEFAULT_TAG = "sem9-hub"


def _get_client() -> Supermemory:
    """Lazy-init the SDK client (reads SUPERMEMORY_API_KEY from env)."""
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = Supermemory()
    return _CLIENT


def memorize(
    content: str,
    container_tag: str = _DEFAULT_TAG,
    metadata: dict[str, object] | None = None,
) -> bool:
    """Persist *content* into Supermemory.

    Returns ``True`` on success, ``False`` if the API key is missing or the
    call fails (so callers can safely fire-and-forget).
    """
    if not os.environ.get("SUPERMEMORY_API_KEY"):
        return False
    try:
        _get_client().add(
            content=content,
            container_tags=[container_tag],
            metadata=metadata or {},
        )
        return True
    except Exception:
        return False


def recall(
    query: str,
    container_tag: str = _DEFAULT_TAG,
    limit: int = 10,
) -> list[dict[str, object]]:
    """Search Supermemory for memories matching *query*.

    Returns a list of result dicts with keys ``document_id``, ``content``,
    ``title``, ``score``, ``chunks``, ``metadata``, ``summary``.
    Returns an empty list when the API key is missing or the call fails.
    """
    if not os.environ.get("SUPERMEMORY_API_KEY"):
        return []
    try:
        resp = _get_client().search.documents(
            q=query,
            container_tags=[container_tag],
        )
        return [r.model_dump() for r in resp.results[:limit]]
    except Exception:
        return []
