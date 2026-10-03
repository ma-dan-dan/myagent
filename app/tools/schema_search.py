from __future__ import annotations

from app.schemas.chat import SearchSchemaInput, SearchSchemaOutput
from app.storage.schema_catalog import SchemaCatalog


class SchemaSearchTool:
    name = "search_schema"

    def __init__(self, catalog: SchemaCatalog) -> None:
        self.catalog = catalog

    def run(self, request: SearchSchemaInput) -> SearchSchemaOutput:
        validated_request = SearchSchemaInput.model_validate(request)
        items = self.catalog.search(validated_request.query, validated_request.limit)
        if not items:
            return SearchSchemaOutput(
                status="empty",
                message="没有匹配的 Schema 元数据，请换一种业务名称。",
                items=[],
            )
        return SearchSchemaOutput(
            status="ok",
            message=f"找到 {len(items)} 个匹配的 Schema。",
            items=items,
        )
