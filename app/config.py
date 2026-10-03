from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "chat.sqlite3"
DEFAULT_CATALOG_PATH = PROJECT_ROOT / "data" / "schema_catalog.json"
DEFAULT_WEB_PATH = PROJECT_ROOT / "web" / "index.html"
