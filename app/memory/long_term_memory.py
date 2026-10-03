from __future__ import annotations

import logging
import re
import tempfile
import unicodedata
from pathlib import Path

from app.memory.models import MemoryEntry


logger = logging.getLogger(__name__)
_CREDENTIAL_RE = re.compile(
    r"api[_ -]?key|password|secret|cookie|authorization|bearer\s|sk-|access_token|refresh_token|api_token",
    re.IGNORECASE,
)
_MARKDOWN_LINK_RE = re.compile(r"\[[^\]]+\]\([^)]*\)|https?://|(?:^|\s)[A-Za-z]:[\\/]|(?:^|\s)[\\/]{1,2}")


class LongTermMemoryStore:
    def __init__(
        self,
        workspace_root: str | Path,
        max_lines: int = 200,
        max_bytes: int = 25_000,
    ) -> None:
        self.root = Path(workspace_root) / ".myagent" / "memory" / "chat"
        self.memory_path = self.root / "MEMORY.md"
        self.max_lines = max_lines
        self.max_bytes = max_bytes

    def load(self) -> str:
        try:
            raw = self.memory_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            return ""
        if not raw.strip():
            return ""
        lines = raw.splitlines(keepends=True)[: self.max_lines]
        content = "".join(lines)
        if len(content.encode("utf-8")) <= self.max_bytes:
            return content
        encoded = content.encode("utf-8")[: self.max_bytes]
        content = encoded.decode("utf-8", errors="ignore")
        newline = content.rfind("\n")
        if newline >= 0:
            content = content[: newline + 1]
        return content

    def as_system_message(self) -> dict[str, str] | None:
        content = self.load()
        if not content:
            return None
        return {"role": "system", "content": f"<long_term_memory>\n{content}\n</long_term_memory>"}

    def upsert(self, entries: list[MemoryEntry]) -> list[MemoryEntry]:
        existing = self._read_existing()
        if existing is None:
            return []
        current = existing if existing else "# 长期记忆\n"
        if not self._within_limits(current):
            return []
        seen = {self._normalize(line[2:]) for line in current.splitlines() if line.startswith("- ")}
        accepted: list[MemoryEntry] = []
        for raw_entry in entries:
            try:
                entry = MemoryEntry.model_validate(raw_entry)
                content = self._validate_content(entry.content)
            except Exception as exc:
                logger.warning("Skipping long-term memory entry: %s", exc)
                continue
            if self._normalize(content) in seen:
                continue
            candidate = f"- {content}\n"
            if not self._within_limits(current + candidate):
                continue
            current += candidate
            seen.add(self._normalize(content))
            accepted.append(MemoryEntry(content=content))
        if not accepted:
            return []
        try:
            self._atomic_write(current)
        except OSError as exc:
            logger.warning("Long-term memory write failed: %s", exc)
            return []
        return accepted

    def _read_existing(self) -> str | None:
        try:
            if not self.memory_path.exists():
                return ""
            return self.memory_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            return None

    def _validate_content(self, content: str) -> str:
        content = content.strip()
        if not content or "\n" in content or "\r" in content:
            raise ValueError("memory content must be one non-empty line")
        if any(unicodedata.category(char).startswith("C") for char in content):
            raise ValueError("control characters are not allowed")
        if _CREDENTIAL_RE.search(content):
            raise ValueError("credential-like content is not allowed")
        if _MARKDOWN_LINK_RE.search(content) or "/" in content or "\\" in content:
            raise ValueError("links and paths are not allowed")
        return content

    @staticmethod
    def _normalize(content: str) -> str:
        return re.sub(r"\s+", " ", content.strip()).lower()

    def _within_limits(self, content: str) -> bool:
        return len(content.splitlines()) <= self.max_lines and len(content.encode("utf-8")) <= self.max_bytes

    def _atomic_write(self, content: str) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=self.root, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
        try:
            temporary.replace(self.memory_path)
        finally:
            if temporary.exists():
                temporary.unlink()
