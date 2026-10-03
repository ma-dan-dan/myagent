import pytest
from pydantic import ValidationError

from app.memory.long_term_memory import LongTermMemoryStore
from app.memory.models import MemoryEntry


def test_long_term_memory_writes_direct_entries_to_single_memory_file(tmp_path):
    store = LongTermMemoryStore(tmp_path)
    entries = [MemoryEntry(content="用户希望默认使用中文解释代码概念。"), MemoryEntry(content="项目测试必须使用 FakeLLM。")]

    assert store.upsert(entries) == entries

    memory_path = tmp_path / ".myagent" / "memory" / "chat" / "MEMORY.md"
    assert memory_path.exists()
    assert not (tmp_path / ".myagent" / "memory" / "chat" / "topics").exists()
    content = memory_path.read_text(encoding="utf-8")
    assert content.startswith("# 长期记忆")
    assert "- 用户希望默认使用中文解释代码概念。" in content
    assert "- 项目测试必须使用 FakeLLM。" in content
    assert store.load() == content
    assert store.as_system_message() == {"role": "system", "content": f"<long_term_memory>\n{content}\n</long_term_memory>"}
    assert not hasattr(store, "retrieve")


def test_long_term_memory_deduplicates_normalized_content(tmp_path):
    store = LongTermMemoryStore(tmp_path)
    first = MemoryEntry(content="  Stable   project rule  ")
    duplicate = MemoryEntry(content="stable project rule")

    assert store.upsert([first]) == [MemoryEntry(content="Stable   project rule")]
    assert store.upsert([duplicate]) == []
    assert store.load().count("Stable") == 1


def test_long_term_memory_rejects_credentials_and_invalid_lines(tmp_path):
    store = LongTermMemoryStore(tmp_path)
    unsafe = [
        MemoryEntry(content="api_key=do-not-store"),
        MemoryEntry(content="Bearer abc123"),
        MemoryEntry(content="access_token=abc123"),
        MemoryEntry(content="first line\nsecond line"),
    ]

    assert store.upsert(unsafe) == []
    with pytest.raises(ValidationError):
        MemoryEntry(content="x" * 301)
    assert not (tmp_path / ".myagent" / "memory" / "chat" / "MEMORY.md").exists()


def test_long_term_memory_skips_entries_that_exceed_injected_capacity(tmp_path):
    store = LongTermMemoryStore(tmp_path, max_lines=3)
    old = MemoryEntry(content="old stable rule")
    new = MemoryEntry(content="new stable rule")
    too_much = MemoryEntry(content="another stable rule")

    assert store.upsert([old]) == [old]
    assert store.upsert([new, too_much]) == [new]
    assert store.load().splitlines() == ["# 长期记忆", "- old stable rule", "- new stable rule"]


def test_long_term_memory_skips_entries_that_exceed_byte_capacity(tmp_path):
    store = LongTermMemoryStore(tmp_path, max_bytes=45)
    old = MemoryEntry(content="old")
    new = MemoryEntry(content="new content that does not fit")

    assert store.upsert([old]) == [old]
    assert store.upsert([new]) == []
    assert store.load() == "# 长期记忆\n- old\n"


def test_long_term_memory_load_limits_are_read_only(tmp_path):
    memory_path = tmp_path / ".myagent" / "memory" / "chat" / "MEMORY.md"
    memory_path.parent.mkdir(parents=True)
    original = "# 长期记忆\n" + "\n".join(f"- {'x' * 100}-{index}" for index in range(250)) + "\n"
    memory_path.write_text(original, encoding="utf-8")

    loaded = LongTermMemoryStore(tmp_path).load()

    assert len(loaded.splitlines()) <= 200
    assert len(loaded.encode("utf-8")) <= 25_000
    assert memory_path.read_text(encoding="utf-8") == original


def test_long_term_memory_missing_file_is_empty_and_does_not_create_directory(tmp_path):
    store = LongTermMemoryStore(tmp_path)

    assert store.load() == ""
    assert store.as_system_message() is None
    assert not (tmp_path / ".myagent" / "memory" / "chat").exists()
