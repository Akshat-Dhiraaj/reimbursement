"""Web UI backend for slipguard.

A thin FastAPI wrapper around the :func:`slipguard.llm_validate.validate` pipeline (LLM judge +
deterministic cross-check). Paired with the React drag-and-drop frontend in ``./frontend`` (Vite).

Run it with either::

    slipguard serve                          # convenience wrapper (needs the [web] extra)
    uvicorn slipguard.web.api:app --reload    # the same app, directly

The package ``__init__`` intentionally does **not** import :mod:`slipguard.web.api`, so importing
``slipguard.web`` never requires FastAPI; the import-guard lives in ``api`` and ``cli serve`` and
tells you to ``pip install -e ".[web]"`` if it is missing.
"""
