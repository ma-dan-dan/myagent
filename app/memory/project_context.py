from __future__ import annotations

from pathlib import Path


class ProjectContextLoader:
    def __init__(self, workspace_root: str | Path, max_lines: int = 200, max_bytes: int = 25_000) -> None:
        self.workspace_root = Path(workspace_root)
        self.max_lines = max_lines
        self.max_bytes = max_bytes

    def load(self) -> str:
        path = self.workspace_root / "Agent.md"
        try:
            raw = path.read_bytes()
        except OSError:
            return ""
        truncated = len(raw) > self.max_bytes
        text = raw[: self.max_bytes].decode("utf-8", errors="replace")
        lines = text.splitlines()[: self.max_lines]
        if len(text.splitlines()) > self.max_lines:
            truncated = True
        result = "\n".join(lines)
        if truncated:
            result += "\n[Agent.md 已截断，以上内容不是完整项目规则。]"
        return result

    def as_system_message(self) -> dict[str, str] | None:
        content = self.load()
        if not content:
            return None
        return {"role": "system", "content": f"<project_rules>{content}</project_rules>"}
