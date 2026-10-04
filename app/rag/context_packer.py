from __future__ import annotations

from app.memory.token_manager import TokenManager
from app.rag.models import SchemaCandidate


class RagContextBudgetExceeded(RuntimeError):
    pass


class ContextPacker:
    def __init__(self, token_manager: TokenManager) -> None:
        self.token_manager = token_manager

    def pack(self, candidates: list[SchemaCandidate], budget: int) -> str:
        parts: list[str] = []
        for candidate in candidates:
            samples = "; ".join(f"{column}: {', '.join(values[:3])}" for column, values in candidate.sample_values.items())
            part = f"<schema_evidence table=\"{candidate.table_name}\">\n{candidate.ddl}\n命中列: {', '.join(candidate.matched_columns)}\n样例: {samples}\n</schema_evidence>"
            proposed = "\n".join(parts + [part])
            if self.token_manager.estimate([{"role": "system", "content": proposed}]) > budget:
                if not parts:
                    raise RagContextBudgetExceeded("Schema evidence exceeds the available context budget.")
                break
            parts.append(part)
        return "\n".join(parts)
