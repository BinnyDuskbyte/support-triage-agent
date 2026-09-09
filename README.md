# Support Triage Agent

An AI service that triages inbound support tickets: it **classifies** the ticket,
**answers from your own help-docs** (RAG) with citations, and then **auto-replies,
escalates, or routes** it — with a human-in-the-loop review step until the eval
scores justify auto-send.

Python 3.12 · FastAPI · OpenAI structured outputs · n8n orchestration

[![ci](https://github.com/BinnyDuskbyte/support-triage-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/BinnyDuskbyte/support-triage-agent/actions/workflows/ci.yml)

---

## The problem

Small SaaS and e-commerce support teams drown in repetitive tickets. First-response
time slips, FAQs eat hours that should go to real issues, and the obvious fix — "just
put a chatbot on it" — fails the moment the bot invents a refund policy that doesn't
exist.

The hard part isn't calling an LLM. It's calling one you can *trust in front of a
customer*. This project is built around that constraint.

## How it works

```
        ┌──────────────┐
inbound │     n8n      │  webhook: email / form / Slack / Zendesk
ticket  │ orchestration│
───────►│              │
        └──────┬───────┘
               │ POST /triage
               ▼
      ┌────────────────────────────────────────────────┐
      │              FastAPI  (this repo)              │
      │                                                │
      │  1. classify   intent · urgency · sentiment    │
      │                → structured output, not prose  │
      │                                                │
      │  2. retrieve   top-k help-doc chunks           │
      │                → embeddings + vector search    │
      │                                                │
      │  3. draft      grounded reply + citations      │
      │                → refuses if context is thin    │
      │                                                │
      │  4. gate       confidence → decision           │
      └───────────────────────┬────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
   auto-reply            escalate              route
   (high conf.)      (human review queue)   (billing / eng / …)
```

The split is deliberate: **reasoning lives in FastAPI** where it can be unit-tested
and evaluated; **orchestration lives in n8n** where a non-developer can rewire which
inbox it listens to and where escalations land.

## Design principles

These are the decisions worth defending in a code review:

| Principle | What it means here |
|---|---|
| **Structured outputs, never regex** | Every LLM call that must return data is schema-constrained via `chat.completions.parse` with a Pydantic model. The schema constrains *decoding*, so the model physically cannot emit an enum value we didn't define or wrap JSON in markdown fences. |
| **Ground, then answer** | Replies are drafted only from retrieved help-doc context, with citations. If the retrieved context doesn't support an answer, the agent says so and escalates — it does not improvise. |
| **Schema-valid ≠ correct** | A forced schema turns "I don't know" into a confident wrong answer unless you leave an escape hatch. Every schema carries an `other`/`unsure` option and a `confidence` field. |
| **Confidence gating + HITL** | Low-confidence drafts go to a review queue, not to the customer. Auto-send is unlocked per-category only once the eval set earns it. |
| **Evals as tests** | A small `tests/eval_set.jsonl` of tickets + expected outcomes runs in CI. A drop in eval score fails the build, the same as a broken unit test. |
| **Errors are typed and caught** | The LLM is a network dependency that fails. Every failure mode (missing key, API error, refusal, off-schema response) is its own catchable exception; the service returns 502/503, never a traceback. |
| **Cost awareness** | Defaults to the smallest model that passes evals (`gpt-4o-mini`), `temperature=0` for extraction. Moving up a tier requires a reason. |

## Project status

Built in vertical slices — each one lands tested. Current state:

| Slice | Scope | Status |
|---|---|---|
| **1** | Project setup, config, typed OpenAI structured-output wrapper | ✅ shipped |
| **2** | `POST /classify` — intent · urgency · sentiment | 🚧 in progress |
| **3** | Help-doc ingestion, chunking, embeddings, retrieval | ⬜ planned |
| **4** | Grounded reply drafting with citations + refusal path | ⬜ planned |
| **5** | Confidence-gated decision logic (auto / escalate / route) | ⬜ planned |
| **6** | n8n workflow: webhook → FastAPI → actions | ⬜ planned |
| **7** | Eval set + human-in-the-loop review queue | ⬜ planned |
| **8** | Docker packaging + deployment notes | ⬜ planned |

What's runnable today is slice 1: the service boots, and `/ping-llm` proves the
end-to-end structured-output loop (prompt in → validated Pydantic object out, with
every failure mode handled).

## Quickstart

Requires Python 3.12+ and an OpenAI API key.

```bash
git clone https://github.com/BinnyDuskbyte/support-triage-agent.git
cd support-triage-agent

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

cp .env.example .env        # then add your key — .env is gitignored
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000/docs for the interactive OpenAPI UI.

```bash
curl http://127.0.0.1:8000/health
# {"status":"ok"}

curl http://127.0.0.1:8000/ping-llm
# {"sentence":"The refund finally arrived, but it took three weeks and four emails.",
#  "result":{"sentiment":"neg","confidence":0.82}}
```

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness probe. No LLM, no network — safe for k8s/uptime checks. |
| `GET` | `/welcome` | Service banner. |
| `GET` | `/ping-llm` | Demo of the structured-output loop on a fixed sentence. Scaffolding; removed once `/classify` lands. |
| `GET` | `/docs` | Swagger UI (FastAPI built-in). |

Planned: `POST /classify`, `POST /retrieve`, `POST /draft`, `POST /triage` (the full pipeline).

**Error contract:** `503` when the service is misconfigured (no API key — the message
names the variable, never its value), `502` when the upstream model call fails or
returns off-schema output.

## Configuration

All configuration comes from the environment; `.env` is loaded at import but never
overrides a real environment variable, so CI and Docker always win.

| Variable | Required | Default | Notes |
|---|---|---|---|
| `OPENAI_API_KEY` | yes | — | Read lazily, never logged or echoed. |
| `OPENAI_MODEL` | no | `gpt-4o-mini` | Smallest model that passes the evals. |

No secret is ever committed, printed, or included in a log line — errors log the
*exception type*, never the payload.

## Using the LLM wrapper

The core primitive is one function. Hand it a Pydantic model, get a validated
instance back or a typed exception:

```python
from typing import Literal
from pydantic import BaseModel
from app.llm import structured, LLMError

class Sentiment(BaseModel):
    sentiment: Literal["pos", "neg", "neutral"]   # an enum in the JSON schema
    confidence: float

try:
    result = structured("Classify: my order never arrived", Sentiment)
except LLMError as exc:      # catches missing key, API failure, refusal, bad schema
    ...
```

The `client` argument is injectable, which is why the test suite never touches the
network.

## Development

```bash
pytest                 # unit tests — no network calls, fully deterministic
ruff check . && black --check .
```

Tests fake the OpenAI client at the boundary and assert *our* contract: valid output
parses, malformed output raises `LLMValidationError`, a missing key raises
`MissingAPIKeyError` with an actionable message. CI runs lint + tests on every push
and PR.

## Layout

```
app/
  main.py      FastAPI app, routes, HTTP error mapping
  config.py    environment/secret loading (lazy, test-friendly)
  llm.py       structured-output wrapper + typed error hierarchy
tests/         unit tests (network-free); eval set lands in slice 7
.github/workflows/ci.yml
```

## License

GPL-3.0 — see [LICENSE](LICENSE).

## Author

Built by **Binny Chanchal** ([DuskByte](https://duskbyte.com)) as a production-shaped
reference implementation of a grounded, gated support agent.
