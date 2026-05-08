"""Configuración local para proveedores de datos opcionales."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path


def get_alpha_vantage_api_key(project_root: str | Path | None = None) -> str | None:
    """Obtiene la API key de Alpha Vantage desde variables de entorno o secrets.

    Orden de búsqueda:
        1. Variable de entorno `ALPHA_VANTAGE_API_KEY`.
        2. Archivo local `.streamlit/secrets.toml`.

    La llave no debe subirse a Git ni escribirse en archivos públicos.
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
