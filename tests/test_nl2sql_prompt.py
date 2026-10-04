from app.nl2sql.prompt import build_gensql_messages, build_reflection_messages


def test_gensql_prompt_contains_question_schema_and_read_only_constraints():
    messages = build_gensql_messages(
        "查询产量",
        "production_output(output_quantity INTEGER, output_date DATE)",
        "sqlite",
        None,
    )
    text = "\n".join(message["content"] for message in messages)

    assert "查询产量" in text
    assert "production_output" in text
    assert "SELECT/WITH" in text
    assert "不能虚构" in text


def test_reflection_prompt_has_sql_and_safe_execution_summary():
    messages = build_reflection_messages(
        "查询产量",
        "production_output(output_quantity INTEGER)",
        "SELECT output_quantity FROM production_output",
        None,
        "columns=output_quantity; row_count=1",
        "sqlite",
    )
    text = "\n".join(message["content"] for message in messages)

    assert "SELECT output_quantity" in text
    assert "row_count=1" in text
    assert "API Key" not in text
    assert "Bearer" not in text
