"""FastAPI entrypoint for the support-triage service.

Run locally with::

    uvicorn app.main:app --reload
"""

import logging
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.llm import LLMError, LLMRateLimitError, MissingAPIKeyError, structured

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


class SentimentResult(BaseModel):
    """The schema that proves the structured-output loop works.

    Two deliberate choices:

    * `sentiment` is a Literal, not a str. That becomes an enum in the JSON
      schema, so "positive" or "happy-ish" cannot come back — only these three.
    * `confidence` carries no ge/le constraint. Pydantic would happily enforce
      one, but OpenAI's strict structured-output mode supports only a subset of
      JSON Schema and rejects `minimum`/`maximum` on numbers. Range checks
      belong in our own code after parsing, not in the wire schema.
    """

    sentiment: Literal["pos", "neg", "neutral"]
    confidence: float


class PingResponse(BaseModel):
    """What `/ping-llm` hands back.

    A model rather than a bare dict because the payload is nested: FastAPI reads
    the handler's return annotation as its response model, and `dict[str, str]`
    cannot describe a `SentimentResult` sitting under `result`.
    """

    sentence: str
    result: SentimentResult


PING_SENTENCE = "The refund finally arrived, but it took three weeks and four emails."


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe. No LLM, no network — just 'the process is up'."""
    return {"status": "ok"}


@app.get("/welcome")
def welcome() -> dict[str, str]:
    """Welcome to Support Triage Agemt"""
    return {"status": "ok", "message": "Welcome to Support Triage Agent"}


@app.get("/ping-llm")
def ping_llm() -> PingResponse:
    """Throwaway endpoint: prove we can get validated JSON out of the model.

    This is scaffolding for upcoming work only. The real classifier lands soon.
    """
    try:
        result = structured(
            f"Classify the sentiment of this customer message: \n\n{PING_SENTENCE}",
            SentimentResult,
        )
    except MissingAPIKeyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except LLMRateLimitError as exc:
        if exc.quota_exhausted:
            # Not the caller's fault and not worth retrying: the account has no
            # credits. Same category as a missing key — the service is not
            # currently configured to do its job.
            logger.error("ping-llm failed: OpenAI quota exhausted (%s)", exc)
            raise HTTPException(
                status_code=503,
                detail="OpenAI quota exhausted — check billing on the API account.",
            ) from exc

        # A real rate limit. Pass 429 through rather than flattening it to 502:
        # the caller can act on "slow down and retry", not on "bad gateway".
        logger.warning("ping-llm rate limited (retry_after=%s)", exc.retry_after)
        raise HTTPException(
            status_code=429,
            detail=str(exc),
            headers={"Retry-After": str(exc.retry_after)} if exc.retry_after else None,
        ) from exc
    except LLMError as exc:
        logger.warning("ping-llm failed: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return PingResponse(sentence=PING_SENTENCE, result=result)
