from app.memory.project_context import ProjectContextLoader


def test_project_context_loader_reads_only_agent_md_and_wraps_rules(tmp_path):
    (tmp_path / "Agent.md").write_text("project rule", encoding="utf-8")

    loader = ProjectContextLoader(tmp_path)

    assert loader.load() == "project rule"
    assert loader.as_system_message() == {"role": "system", "content": "<project_rules>project rule</project_rules>"}


def test_project_context_loader_returns_empty_when_agent_md_missing(tmp_path):
    loader = ProjectContextLoader(tmp_path)

    assert loader.load() == ""
    assert loader.as_system_message() is None
