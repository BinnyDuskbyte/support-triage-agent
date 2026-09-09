"""Configuration loaded from the environment.

Secrets live in `.env` (gitignored), never in code. `load_dotenv()` copies that
file into `os.environ` at import time; it does NOT overwrite variables that are
already set, so a real environment variable (CI, Docker, systemd) always wins
over the local file.

The API key is read *lazily* inside `get_api_key()` rather than captured into a
module-level constant. That matters: a module constant would freeze whatever the
environment looked like at import time, and tests that manipulate the
environment (`monkeypatch.delenv`) could never take effect.
"""

import os

from dotenv import load_dotenv

load_dotenv()

# The smallest model that passes our evals. Cost/latency awareness starts here:
# every task should justify moving up a tier, never default to the biggest model.
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def get_api_key() -> str | None:
    """Return the OpenAI API key, or None if it is absent/blank.

    Never log or echo the return value of this function.
    """
    key = os.getenv("OPENAI_API_KEY", "").strip()
    return key or None
