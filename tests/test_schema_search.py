import pytest
from pydantic import ValidationError

from app.schemas.chat import SearchSchemaInput
from app.storage.schema_catalog import SchemaCatalog
from app.tools.schema_search import SchemaSearchTool


def test_search_schema_matches_metadata_case_insensitively(catalog_path):
    tool = SchemaSearchTool(SchemaCatalog(catalog_path))

    result = tool.run(SearchSchemaInput(query="OUTPUT", limit=5))

    assert result.status == "ok"
    assert result.items
    assert result.items[0].table_name == "production_output"
    assert result.items[0].columns[0].column_name == "output_date"
    assert not hasattr(result.items[0], "rows")


def test_search_schema_rejects_blank_query_and_invalid_limit():
    with pytest.raises(ValidationError):
        SearchSchemaInput(query=" ", limit=5)

    with pytest.raises(ValidationError):
        SearchSchemaInput(query="产量", limit=0)


def test_search_schema_empty_result_is_structured(catalog_path):
    tool = SchemaSearchTool(SchemaCatalog(catalog_path))

    result = tool.run(SearchSchemaInput(query="不存在的业务", limit=5))

    assert result.status == "empty"
    assert result.items == []
    assert "没有匹配" in result.message
