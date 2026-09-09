import logging

from openai import OpenAI, OpenAIError, RateLimitError
from pydantic import BaseModel, ValidationError

from app.config import DEFAULT_MODEL, get_api_key

logger = logging.getLogger(__name__)


DEFAULT_SYSTEM_PROMPT = (
    "You are a precise extraction engine. Answer only with data that conforms to "
    "the provided schema. If the input does not support a confident answer, say so "
    "through the schema's own fields rather than guessing."
)


class LLMError(RuntimeError):
    """Base class for every failure raised by this module."""


class MissingAPIKeyError(LLMError):
    """No usable OPEN_API_KEY was found in the environment."""


class LLMCallError(LLMError):
    """The API call itself failed: network, auth, timeout."""


class LLMRateLimitError(LLMCallError):
    """The API answered 429.

    Two very different situations share that status code, and a caller has to
    tell them apart before deciding what to do:

    * `quota_exhausted=False` — a genuine rate limit (requests or tokens per
      minute). Transient: waiting `retry_after` seconds and trying again works.
    * `quota_exhausted=True` — the account is out of credits
      (`insufficient_quota`). Retrying never helps; someone has to fix billing.

    Note that by the time this is raised the SDK has already retried on its own
    (`max_retries` defaults to 2), so a single one of these represents several
    attempts on the wire.
    """

    def __init__(
        self,
        message: str,
        *,
        quota_exhausted: bool = False,
        retry_after: int | None = None,
    ) -> None:
        super().__init__(message)
        self.quota_exhausted = quota_exhausted
        self.retry_after = retry_after


class LLMRefusalError(LLMError):
    """The model declined to answer (safety refusal) instead of returning data."""


class LLMValidationError(LLMError):
    """The model's answer did not conform to the requested schema."""


def _retry_after_seconds(exc: RateLimitError) -> int | None:
    """How long the API asked us to wait, if it said anything at all.

    OpenAI sends `retry-after` in seconds and sometimes `retry-after-ms`. Any
    value we cannot parse is treated as absent: no hint beats a wrong hint.
    """
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if headers is None:
        return None

    for name, divisor in (("retry-after-ms", 1000), ("retry-after", 1)):
        raw = headers.get(name)
        if not raw:
            continue
        try:
            return max(1, round(float(raw) / divisor))
        except ValueError:
            continue

    return None


def build_client() -> OpenAI:
    api_key = get_api_key()
    if api_key is None:
        raise MissingAPIKeyError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key. "
            "The .env file is gitignored — never commit it."
        )
    return OpenAI(api_key=api_key)


def structured[SchemaT: BaseModel](
    prompt: str,
    schema: type[SchemaT],
    *,
    client: OpenAI | None = None,
    model: str = DEFAULT_MODEL,
    system: str = DEFAULT_SYSTEM_PROMPT,
    temperature: float = 0.0,
) -> SchemaT:
    """Ask the model for `prompt` and return a validated instance of `schema`.

    `client` is injectable so tests can pass a stand-in and never touch the
    network. Note that we only build a real client — and therefore only require
    an API key — when the caller did not supply one.

    temperature=0 because this is extraction, not writing: we want the same
    input to produce the same structured answer run after run.

    Raises:
        MissingAPIKeyError: no key configured and no client injected.
        LLMRateLimitError: the API answered 429 (rate limit, or out of quota).
        LLMCallError: the API call failed for any other reason.
        LLMRefusalError: the model refused to answer.
        LLMValidationError: the response did not match `schema`.
    """
    client = client or build_client()

    try:
        completion = client.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            response_format=schema,
            temperature=temperature,
        )

    except RateLimitError as exc:
        # `code` comes off the response body: "insufficient_quota" means billing,
        # anything else means we are simply going too fast.
        quota_exhausted = getattr(exc, "code", None) == "insufficient_quota"
        logger.warning("OpenAI rate limited (quota_exhausted=%s): %s", quota_exhausted, exc)
        raise LLMRateLimitError(
            str(exc),
            quota_exhausted=quota_exhausted,
            retry_after=_retry_after_seconds(exc),
        ) from exc

    except OpenAIError as exc:
        # Keep the API's own message: the exception class alone ("APIStatusError")
        # is never enough to tell what actually went wrong.
        logger.warning("OpenAI call failed (%s): %s", type(exc).__name__, exc)
        raise LLMCallError(f"OpenAI call failed ({type(exc).__name__}): {exc}") from exc

    message = completion.choices[0].message

    if getattr(message, "refusal", None):
        raise LLMRefusalError(f"Model refused to answer: {message.refusal}")

    parsed = getattr(message, "parsed", None)
    if parsed is not None:
        return parsed

    raw = getattr(message, "content", None)

    if not raw:
        raise LLMValidationError(
            f"Model returned no parsable content for schema {schema.__name__}. "
            "The response was likely truncated — check max_tokens."
        )

    try:
        return schema.model_validate_json(raw)
    except ValidationError as exc:
        raise LLMValidationError(
            f"Model response did not match schema {schema.__name__}: {exc}"
        ) from exc
