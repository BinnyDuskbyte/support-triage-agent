"""FastAPI entrypoint for the support-triage service.

Run locally with::

    uvicorn app.main:app --reload
"""

import logging

from fastapi import FastAPI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s, %(levelname)s, %(name)s, %(message)s, %(filename)s:%(lineno)d",
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Support Triage Agent",
    version="0.1.0",
    description="Classify -> retrieve -> draft -> gate. Build 1.",
)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe. No LLM, no network — just 'the process is up'."""
    return {"status": "ok"}


@app.get("/welcome")
def welcome() -> dict[str, str]:
    """Welcome to Support Triage Agemt"""
    return {"status": "ok", "message": "Welcome to Support Triage Agent"}


@app.get("/ping-llm")
def ping_llm() -> dict[str, str]:
    """Throwaway endpoint: prove we can get validated JSON out of the model."""
    return {"status": "ok", "message": "LLM Ping Here."}
