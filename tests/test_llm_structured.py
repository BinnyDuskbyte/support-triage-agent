"""Tests for the llm.structured() wrapper.

No test here talks to the real OpenAI API. We fake the client at the boundary,
mimicking only the response shape our wrapper reads. That keeps tests fast,
free, and deterministic — and pins down OUR contract:

  happy path  -> a validated instance of the caller's schema
  refusal     -> LLMRefusalError, not a confusing validation failure
  truncation  -> LLMValidationError naming the real cause
  bad output  -> LLMValidationError (clear, catchable — never a crash)
  no API key  -> MissingAPIKeyError with an actionable message

The fakes are built from the SDK's own response types rather than loose
`SimpleNamespace` stand-ins. That matters more under the Responses API than it
did under Chat Completions: a refusal is no longer a flat `message.refusal`
field but a content part nested inside an output item, and hand-rolled
duck-typing of that nesting would happily pass while the real shape drifts out
from under us. Constructing the real objects means these tests fail if our
reading of the SDK is wrong.
"""

from typing import Literal

import pytest
from openai.types.responses import (
    ParsedResponse,
    ParsedResponseOutputMessage,
    ParsedResponseOutputText,
    ResponseOutputRefusal,
)
from openai.types.responses.response import IncompleteDetails
from pydantic import BaseModel

from app.llm import (
    LLMRefusalError,
    LLMValidationError,
    MissingAPIKeyError,
    structured,
)


class Sentiment(BaseModel):
    sentiment: Literal["pos", "neg", "neutral"]
    confidence: float


def _text(payload: str, parsed: BaseModel | None = None) -> ParsedResponseOutputText:
    """One `output_text` content part. `parsed` is what the SDK managed to
    deserialize — None when the model's text did not fit the schema."""
    return ParsedResponseOutputText(type="output_text", text=payload, annotations=[], parsed=parsed)


def _message(*parts) -> ParsedResponseOutputMessage:
    return ParsedResponseOutputMessage(
        id="msg_1",
        role="assistant",
        status="completed",
        type="message",
        content=list(parts),
    )


def _response(*output, status: str = "completed", incomplete_details=None) -> ParsedResponse:
    """A ParsedResponse carrying `output`, with the required scaffolding filled in."""
    return ParsedResponse(
        id="resp_1",
        created_at=0,
        model="gpt-4o-mini",
        object="response",
        output=list(output),
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
        status=status,
        incomplete_details=incomplete_details,
    )


class _FakeClient:
    """Stand-in for openai.OpenAI exposing just `client.responses.parse(...)`.

    Records the kwargs it was called with so a test can assert on what we sent,
    not only on what we did with the answer.
    """

    def __init__(self, response: ParsedResponse) -> None:
        self.last_kwargs: dict | None = None
        outer = self

        class _Responses:
            def parse(self, **kwargs):
                outer.last_kwargs = kwargs
                return response

        self.responses = _Responses()


def test_structured_output_validates():
    parsed = Sentiment(sentiment="pos", confidence=0.93)
    client = _FakeClient(_response(_message(_text(parsed.model_dump_json(), parsed))))

    result = structured("Classify: I love this!", Sentiment, client=client)

    assert isinstance(result, Sentiment)
    assert result.sentiment == "pos"
    assert result.confidence == 0.93


def test_request_uses_responses_api_shape():
    """The prompt and schema reach the API as Responses-style arguments.

    Worth asserting explicitly: `instructions`/`input`/`text_format` are the
    Responses spellings of what used to be `messages`/`response_format`, and a
    silent regression to the old names would fail against the live API only.
    """
    parsed = Sentiment(sentiment="neutral", confidence=0.5)
    client = _FakeClient(_response(_message(_text(parsed.model_dump_json(), parsed))))

    structured("Classify: meh.", Sentiment, client=client, system="Be terse.")

    sent = client.last_kwargs
    assert sent["input"] == "Classify: meh."
    assert sent["instructions"] == "Be terse."
    assert sent["text_format"] is Sentiment


def test_store_is_off_by_default():
    """Support tickets carry customer data, so we opt out of server-side
    retention — the opposite of the API's own default. Regressing this would be
    silent and invisible in the response, hence a test."""
    parsed = Sentiment(sentiment="neg", confidence=0.7)
    client = _FakeClient(_response(_message(_text(parsed.model_dump_json(), parsed))))

    structured("Classify: awful.", Sentiment, client=client)

    assert client.last_kwargs["store"] is False


def test_refusal_raises_refusal_error():
    """A refusal must not masquerade as a schema problem: the caller's fix for
    'the model declined' is nothing like their fix for 'the JSON was wrong'."""
    refusal = ResponseOutputRefusal(type="refusal", refusal="I can't help with that.")
    client = _FakeClient(_response(_message(refusal)))

    with pytest.raises(LLMRefusalError, match="can't help"):
        structured("Do something disallowed", Sentiment, client=client)


def test_truncated_response_names_the_real_cause():
    """When the model runs out of room the API says so outright, so our error
    should point at the token budget rather than blame the schema."""
    client = _FakeClient(
        _response(
            status="incomplete",
            incomplete_details=IncompleteDetails(reason="max_output_tokens"),
        )
    )

    with pytest.raises(LLMValidationError, match="max_output_tokens"):
        structured("Classify: I love this!", Sentiment, client=client)


def test_malformed_response_raises_clear_validation_error():
    # The SDK couldn't parse (parsed=None) and the raw text is off-schema:
    # "positive" is not in the enum, "high" is not a float.
    malformed = '{"sentiment": "positive", "confidence": "high"}'
    client = _FakeClient(_response(_message(_text(malformed, parsed=None))))

    with pytest.raises(LLMValidationError, match="Sentiment"):
        structured("Classify: I love this!", Sentiment, client=client)


def test_empty_output_raises_validation_error():
    client = _FakeClient(_response())

    with pytest.raises(LLMValidationError, match="no parsable content"):
        structured("Classify: I love this!", Sentiment, client=client)


def test_no_key_fails_clearly(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(MissingAPIKeyError, match="OPENAI_API_KEY"):
        structured("Classify: I love this!", Sentiment)
