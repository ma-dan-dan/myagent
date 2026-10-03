from __future__ import annotations

from typing import Any

from app.agent.adapter import ToolSpec
from app.schemas.chat import SearchSchemaOutput
from app.tools.schema_search import SchemaSearchTool


class ToolRegistry:
    """Explicit allow-list for the only V1 tool."""

    def __init__(self, tools: list[SchemaSearchTool]) -> None:
        names = {tool.name for tool in tools}
        if names != {"search_schema"}:
            raise ValueError("V1 registry must contain only search_schema")
        self._tools = {tool.name: tool for tool in tools}

    def specs(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                name="search_schema",
                description="Search de-identified table and column metadata; never returns rows.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "maxLength": 200},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 10},
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            )
        ]

    def execute(self, name: str, arguments: dict[str, Any]) -> SearchSchemaOutput:
        if name not in self._tools:
            raise ValueError("unknown tool")
        return self._tools[name].run(arguments)
