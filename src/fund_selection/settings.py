"""Local configuration helpers for optional data providers."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path


def get_alpha_vantage_api_key(project_root: str | Path | None = None) -> str | None:
    """Return Alpha Vantage API key from env vars or Streamlit secrets.

    Search order:
        1. Environment variable `ALPHA_VANTAGE_API_KEY`.
        2. Local `.streamlit/secrets.toml` file.

    The key should never be committed to Git or written into README files.
    """

    env_value = os.getenv("ALPHA_VANTAGE_API_KEY")
    if env_value:
        return env_value.strip()

    root = Path(project_root) if project_root else Path.cwd()
    secrets_path = root / ".streamlit" / "secrets.toml"
    if not secrets_path.exists():
        return None

    with secrets_path.open("rb") as file:
        secrets = tomllib.load(file)

    key = secrets.get("ALPHA_VANTAGE_API_KEY")
    return str(key).strip() if key else None
