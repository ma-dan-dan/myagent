# Minimal Agentic Chat V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a small FastAPI chat application with native HTML, user/session-isolated SQLite history, one Pydantic-validated `search_schema` tool, a bounded injectable LLM tool loop, and fully fake-model automated tests.

**Architecture:** `api/chat.py` validates HTTP input and delegates to `ChatService`. `ChatService` owns request orchestration and composes `SessionService` with `ChatAgent`; the agent owns only the model/tool loop, while `SchemaSearchTool` owns catalog lookup and `SchemaCatalog` owns metadata loading. The adapter boundary is a protocol so tests use a deterministic fake and the default app does not require a real model API.

**Tech Stack:** Python 3.11+, FastAPI, Pydantic v2, SQLite, pytest, httpx, native HTML/CSS/JavaScript.

---

### Task 1: Add failing contract and storage tests

**Files:**
- Create: `tests/conftest.py`
- Create: `tests/test_schema_search.py`
- Create: `tests/test_session_service.py`

- [ ] Write tests for case-insensitive schema lookup, Pydantic rejection of invalid tool input, and structured empty results.
- [ ] Write tests proving messages are persisted for one `user_id + session_id` and are invisible to another user with the same session ID.
- [ ] Run `pytest tests/test_schema_search.py tests/test_session_service.py -q`; expect import failures because `app` does not exist yet.

### Task 2: Implement schema contracts and SQLite storage

**Files:**
- Create: `app/__init__.py`, `app/schemas/__init__.py`, `app/schemas/chat.py`
- Create: `app/storage/__init__.py`, `app/storage/schema_catalog.py`, `app/storage/session_service.py`
- Create: `data/schema_catalog.json`

- [ ] Define Pydantic request/response models for chat, tool input/output, tool events, and stored messages.
- [ ] Implement catalog loading from the small de-identified metadata JSON and keyword matching over table/column names and comments only.
- [ ] Implement SQLite schema keyed by `(user_id, session_id)`, with explicit validation of safe identifiers and bounded history reads.
- [ ] Run the Task 1 tests; expect them to pass.

### Task 3: Add the failing agent-loop tests

**Files:**
- Create: `tests/test_chat_agent.py`

- [ ] Add a fake adapter that returns a direct answer and assert no tool event is produced.
- [ ] Add a fake adapter sequence of tool call then final answer and assert Pydantic-validated input/output, the tool result is sent back, and the final answer is returned.
- [ ] Add empty-result, missing-information follow-up, and tool/agent limit tests.
- [ ] Run `pytest tests/test_chat_agent.py -q`; expect import failures because the adapter and agent do not exist.

### Task 4: Implement the injectable adapter and bounded agent

**Files:**
- Create: `app/agent/__init__.py`, `app/agent/adapter.py`, `app/agent/tool_registry.py`, `app/agent/chat_agent.py`
- Create: `app/tools/__init__.py`, `app/tools/schema_search.py`

- [ ] Define `LLMAdapter.complete(messages, tools)` and validated `LLMResponse` variants for `message` and `tool_call`.
- [ ] Register exactly one tool name, `search_schema`; reject unknown tool names and malformed arguments before execution.
- [ ] Implement a loop capped at two agent rounds and two tool calls, return a stable explanatory failure message on exhaustion, and keep follow-up questions as ordinary assistant text.
- [ ] Run the Task 3 tests; expect them to pass.

### Task 5: Add the service and API tests first

**Files:**
- Create: `tests/test_chat_api.py`

- [ ] Test a first request creates a session and a second request with that ID receives prior context through the service.
- [ ] Test same session ID across different users stays isolated.
- [ ] Test JSON validation errors and stable model/tool errors are returned without stack traces.
- [ ] Run `pytest tests/test_chat_api.py -q`; expect import failures because the app assembly does not exist.

### Task 6: Implement service, route, app assembly, and native page

**Files:**
- Create: `app/services/__init__.py`, `app/services/chat_service.py`
- Create: `app/api/__init__.py`, `app/api/chat.py`
- Create: `app/config.py`, `app/main.py`
- Create: `web/index.html`
- Create: `requirements.txt`

- [ ] Make `ChatService` create/locate sessions, read bounded history, call the agent, and persist the user/assistant/tool trace as one request.
- [ ] Keep the route limited to Pydantic HTTP handling and return only `session_id`, `message`, and safe `tool_events`.
- [ ] Mount `/api/v1/chat` and serve `/` with the native page; inject an adapter through `create_app(llm_adapter=...)` and use an unavailable adapter by default instead of making tests or startup call a real API.
- [ ] Run all tests; expect them to pass.

### Task 7: Final verification and scope audit

**Files:**
- Modify only files needed by failing tests or verification.

- [ ] Run `pytest -q` and record the actual result.
- [ ] Run `rg -n "SQL|Workflow|SSE|MCP|Skill|sub.?agent|shell|file tool|duckdb|lancedb" app tests web data`; inspect matches so only explanatory text or required endpoint names remain.
- [ ] Check every new source comment uses `//` only if comments are added; preserve all existing comments.
- [ ] Report files, test result, and the mapping to the seven DA reference files without claiming unsupported production-model behavior.
