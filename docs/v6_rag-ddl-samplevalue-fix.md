# V6 DDL + SampleValue RAG Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复 V6 RAG 的伪造 SampleValue、敏感值泄露、低分误召回、上下文预算失败和 `data_operation/read` 链路未记录 usage/记忆维护的问题。

**Architecture:** `data_operation/read` 仍然走 `IntentRouter → SchemaLinkingService → DDL/SampleValue 双路召回 → Fusion/ContextPacker → ContextManager → ChatAgent(tools=[])`。SchemaLinkingService 只消费调用方提供的 `RagSchemaRecord`，默认应用不得为字段生成伪造样例；真实样例为空时只建立 DDL 索引。RAG 成功回答后的 usage、滚动摘要和 MEMORY.md 维护必须与普通 chat 使用同一套 ChatService 收尾逻辑。

**Tech Stack:** Python 3、FastAPI、Pydantic、LiteLLM Embedding、Qwen `text-embedding-v3`、LanceDB、SQLite、pytest。

---

## 约束

1. 保留 V6 的 DDL/SampleValue 双路向量召回和 `tools=[]` 主模型单次回答。
2. 不实现 SQL、NL2SQL、真实数据库读写、Workflow、SSE、MCP、Skill、向量重排模型、子 Agent 或新的模型 Provider。
3. 不删除已有注释，不回滚其他提交，不执行 `git reset`、`git checkout` 或删除无关文件。
4. 所有测试使用 FakeEmbedding、FakeLLM 或 mock；不得访问真实模型/Embedding 网络，不写入真实 Key。
5. 运行时环境变量仍只能由 `app/config.py` 读取；本计划不得把 Key 写入代码、测试、README 或索引文件。
6. 完成后必须提交全部相关改动，不得只提交新增文件。

## 当前 Review 发现

- `app/main.py` 为每个列生成 `sample-<column>`，这不是 SampleValue，且会污染召回证据。
- `app/rag/indexer.py` 没有拒绝 password、secret、Key、Cookie、Bearer、Token 等敏感值。
- `SchemaRetriever` 没有低相似度阈值，不相关查询也会返回 LanceDB 的某个 top-k 结果。
- `ContextPacker` 可能返回空上下文，但 `SchemaLinkingService` 仍返回 `status="ok"` 并调用主模型。
- `data_operation/read` 的 `_read_response()` 没有复用普通 chat 的 usage、摘要和 MEMORY.md 写入收尾逻辑。
- `FIXED_SYSTEM_PROMPT` 仍声称只能使用 `search_schema`，与 V6 RAG 上下文链路冲突。
- LanceDB 索引没有清理已从输入记录中移除的旧文档，且过滤条件直接拼接字符串。
- 当前测试没有覆盖上述关键失败场景；RAG 主链路测试主要注入 Fake SchemaLinkingService。

## 文件边界

| 文件 | 本次职责 |
| --- | --- |
| `app/main.py` | 不再制造 SampleValue；支持注入 RAG 记录和测试用索引目录；保持默认 DDL 来源。 |
| `app/storage/schema_catalog.py` | 如确有需要，增加只读公开表访问接口，避免 `main.py` 依赖 `_tables`。 |
| `app/rag/indexer.py` | SampleValue 安全过滤、去重、截断；只索引调用方提供的样例。 |
| `app/rag/models.py` | 为召回/打包状态增加明确模型字段，保持现有兼容字段。 |
| `app/rag/vector_store.py` | namespace 安全过滤、按 namespace 同步并清理陈旧文档。 |
| `app/rag/retriever.py` | 输入校验、低分命中过滤、稳定的 empty/ambiguous 判断。 |
| `app/rag/context_packer.py` | 明确预算不足状态，禁止空证据伪装成成功。 |
| `app/rag/service.py` | 传播 empty/ambiguous/budget_exceeded，不能把无证据结果标成 ok。 |
| `app/memory/context_manager.py` | 更新与 RAG 一致的固定 Prompt；固定上下文超预算时沿用 413 语义，不能静默放行。 |
| `app/services/chat_service.py` | 抽取普通 chat/RAG 共用收尾，确保 usage、摘要和 MEMORY.md 一致。 |
| `app/api/chat.py` | 仅在需要新领域异常时增加稳定的 413/503 映射，保留现有协议。 |
| `tests/test_rag_components.py` 或新建 `tests/test_rag_fix.py` | 先写失败测试，覆盖本计划新增行为。 |
| `tests/test_chat_api.py` | 增加 RAG 主链路、usage/maintenance 和预算错误回归测试。 |
| `README.md` | 只在实际行为变更后补充“样例由外部记录提供、默认不伪造样例”的说明。 |

不修改 `ChatAgent` 的既有 `LLMAdapter.complete(messages, tools)` 契约、IntentClassifier、ToolRegistry 的通用接口、SQLite 原始消息删除策略、LLM Provider 配置或前端。

---

## Task 1：先锁定默认数据源和 SampleValue 安全策略

**Files:**

- Create: `tests/test_rag_fix.py`
- Modify: `tests/test_chat_api.py`
- Modify: `app/main.py`
- Modify: `app/storage/schema_catalog.py`（仅在需要公开只读访问时）
- Modify: `app/rag/indexer.py`

- [ ] **Step 1: 写失败测试：默认应用不制造样例**

为 `create_app` 增加可注入的 `rag_records` 和 `rag_index_path` 参数，测试传入一条没有 `sample_values` 的记录，触发一次索引后验证 SampleValue 索引没有文档；测试不得依赖 `catalog._tables`。

```python
def test_default_rag_records_do_not_fabricate_sample_values(tmp_path):
    records = [
        RagSchemaRecord(
            table_id="orders",
            table_name="orders",
            ddl="CREATE TABLE orders (order_id TEXT);",
            columns=[RagColumnRecord(column_name="order_id", data_type="text")],
        )
    ]
    service = build_schema_linking_service_with_fake_embedding(
        tmp_path / "rag", records
    )
    service.search("order")

    assert service.retriever.vector_store.search(
        RagSourceType.SAMPLE_VALUE, [1.0, 0.0], "default", 10
    ) == []
```

测试还要验证：当调用方显式传入 `sample_values=["ORD-001"]` 时，才生成 SampleValue 文档。

- [ ] **Step 2: 运行失败测试**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_rag_fix.py::test_default_rag_records_do_not_fabricate_sample_values -q
```

预期当前实现失败，因为 `app/main.py` 会为每列生成 `sample-<column>`。

- [ ] **Step 3: 最小实现**

1. 删除默认 `sample_values=[f"sample-{column.column_name}"]` 的行为；默认记录只能使用实际传入的样例，若没有样例则使用空列表。
2. `create_app` 增加 `rag_records: list[RagSchemaRecord] | None = None` 和 `rag_index_path: str | Path | None = None`，默认记录由 catalog 生成 DDL、样例为空；测试可以注入合成记录和临时索引目录。
3. 如需访问 catalog 表，给 `SchemaCatalog` 增加只读 `tables` 属性，不能让业务代码依赖 `_tables`。
4. 保持 DDL 来源为脱敏 Schema 元数据，不添加真实业务数据。

- [ ] **Step 4: 写失败测试：敏感样例拒绝入索引**

```python
@pytest.mark.parametrize("column_name", [
    "password", "api_key", "secret_token", "session_cookie", "authorization"
])
def test_sensitive_sample_values_are_not_indexed(tmp_path, column_name):
    records = [RagSchemaRecord(
        table_id="accounts",
        table_name="accounts",
        ddl="CREATE TABLE accounts (value TEXT);",
        columns=[RagColumnRecord(
            column_name=column_name,
            data_type="text",
            sample_values=["super-secret", "Bearer abc", "token-value"],
        )],
    )]
    indexer = SchemaIndexer(FakeEmbedding([1.0, 0.0]), LanceVectorStore(tmp_path))
    indexer.index(records)

    assert LanceVectorStore(tmp_path).search(
        RagSourceType.SAMPLE_VALUE, [1.0, 0.0], "default", 10
    ) == []
```

同时测试普通非敏感样例会保留，最多 3 个去重值且每个值最多 80 个字符。

- [ ] **Step 5: 实现安全过滤并验证**

在 `SchemaIndexer` 内使用明确的大小写不敏感字段名规则拒绝 `password/passwd/secret/api_key/token/cookie/authorization/bearer/credential` 等字段；对值内容也拒绝包含 `Bearer `、JWT 三段格式或明显 Key/Token 前缀的样例。拒绝时直接跳过该 SampleValue 文档，不把原始值改写后继续存储。DDL 仍按计划保存结构信息，不索引完整原始行。

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_rag_fix.py -k "fabricate or sensitive" -q
```

- [ ] **Step 6: 提交 Task 1**

```powershell
git add app/main.py app/storage/schema_catalog.py app/rag/indexer.py tests/test_rag_fix.py tests/test_chat_api.py
git commit -m "fix: remove fabricated and sensitive rag samples"
```

---

## Task 2：修复向量索引同步和过滤安全

**Files:**

- Modify: `app/rag/vector_store.py`
- Modify: `app/rag/indexer.py`
- Modify: `tests/test_rag_fix.py`

- [ ] **Step 1: 写失败测试：删除陈旧文档**

```python
def test_index_sync_removes_documents_missing_from_new_records(tmp_path):
    store = LanceVectorStore(tmp_path)
    indexer = SchemaIndexer(FakeEmbedding([1.0, 0.0]), store)
    old = make_record("old_table", "old_col", ["OLD"])
    new = make_record("new_table", "new_col", ["NEW"])

    indexer.index([old, new])
    indexer.index([new])

    hits = store.search(RagSourceType.SAMPLE_VALUE, [1.0, 0.0], "default", 10)
    assert [hit.document.table_id for hit in hits] == ["new_table"]
```

- [ ] **Step 2: 写失败测试：特殊 namespace/document id 不破坏查询**

使用包含单引号的合成 namespace/table id 验证 upsert、同步和 search 不产生 LanceDB 过滤语法错误。

- [ ] **Step 3: 最小实现**

1. 为 VectorStore 增加按 `source_type + namespace` 的同步接口，例如 `replace_namespace(source_type, namespace, entries)`。
2. Indexer 每次为某个 source 写入前，先在内存完成 embedding，再用同步接口删除该 namespace 中不在本次文档集合的旧 id，并写入新集合；空集合也必须清理旧文档。
3. 所有 namespace、document id 过滤字符串统一经过 LanceDB 字符串转义；不能直接插入未转义值。
4. 继续保持 DDL 与 SampleValue 两张逻辑表分离。

- [ ] **Step 4: 验证并提交**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_rag_fix.py -k "stale or namespace" -q
git add app/rag/vector_store.py app/rag/indexer.py tests/test_rag_fix.py
git commit -m "fix: synchronize rag vector namespaces safely"
```

---

## Task 3：增加低分过滤和稳定的 empty/ambiguous 结果

**Files:**

- Modify: `app/rag/models.py`
- Modify: `app/rag/retriever.py`
- Modify: `tests/test_rag_fix.py`

- [ ] **Step 1: 写失败测试**

覆盖以下行为：

```python
def test_low_similarity_hits_return_empty():
    store = FakeVectorStore([
        VectorHit(document=ddl_document(), score=0.01, rank=1)
    ])
    result = SchemaRetriever(
        FakeEmbedding([1.0, 0.0]), store, min_hit_score=0.2
    ).search("完全无关的问题", "default", 3)
    assert result.status == "empty"
    assert result.candidates == []


def test_close_top_candidates_return_ambiguous():
    result = build_retriever_with_two_close_candidates().search("业务对象", "default", 3)
    assert result.status == "ambiguous"
    assert len(result.candidates) >= 2
```

还要覆盖 `limit < 1`、空 query 和只命中 DDL/只命中 SampleValue。

- [ ] **Step 2: 最小实现**

1. `SchemaRetriever` 增加显式的 `min_hit_score` 和 `ambiguity_margin` 配置，默认值写在构造函数中，不读取环境变量。
2. 在融合前丢弃低于 `min_hit_score` 的 `VectorHit`；若两路都被过滤，返回 `RetrievalResult(status="empty")`。
3. 对输入 query/limit 做稳定校验；不得因为 LanceDB 返回 top-k 就把任意低相关候选标成 ok。
4. 保留 RRF；候选排序使用 `(-score, table_id)`，ambiguous 只基于排序后的前两名相对差值和配置 margin。
5. 如果模型字段增加 `ddl_score`、`sample_value_score`，必须给默认值并更新现有测试，不能删除现有 `score`、`matched_by` 字段。

- [ ] **Step 3: 验证并提交**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_rag_fix.py -k "similarity or ambiguous or limit" -q
git add app/rag/models.py app/rag/retriever.py tests/test_rag_fix.py
git commit -m "fix: reject low confidence rag matches"
```

---

## Task 4：修复上下文预算和 RAG 失败状态

**Files:**

- Modify: `app/rag/context_packer.py`
- Modify: `app/rag/service.py`
- Modify: `app/memory/context_manager.py`
- Modify: `app/api/chat.py`（只在异常类型需要时）
- Modify: `tests/test_rag_fix.py`
- Modify: `tests/test_chat_api.py`

- [ ] **Step 1: 写失败测试：证据无法打包时不能继续回答**

```python
def test_context_pack_budget_exceeded_is_not_reported_as_ok():
    service = build_schema_linking_service_with_oversized_ddl()

    with pytest.raises(RagContextBudgetExceeded):
        service.search("oversized schema")
```

API 测试验证该异常映射为 HTTP 413，且 FakeLLM 没有被调用、当前请求不会写入 assistant 伪回答。

- [ ] **Step 2: 最小实现**

推荐保留 `ContextPacker.pack(...) -> str` 兼容调用方式，并新增明确异常 `RagContextBudgetExceeded`：

1. 候选列表为空时返回空字符串，由 Service 保持 `empty`。
2. 第一条候选无法放入传入预算时抛出 `RagContextBudgetExceeded`，不能返回 `status="ok"`。
3. 部分候选成功时只返回已纳入的候选；若需要记录截断，使用结构化结果，不把空字符串伪装成 ok。
4. `SchemaLinkingService.search()` 检查 packed context，`ok` 必须保证 context 非空。
5. ContextManager 不得静默放行超过 `fixed_context_budget` 的 RAG fixed context；在无法通过已有摘要压缩解决时抛出 `ContextBudgetExceeded`，交给现有 413 处理。
6. 更新 `FIXED_SYSTEM_PROMPT`：说明模型可以使用注入的 Schema evidence 回答；仍然禁止 SQL、数据行和未注册工具，不再写“只能使用 search_schema”。

- [ ] **Step 3: 验证并提交**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_rag_fix.py tests/test_chat_api.py -k "budget or context or evidence" -q
git add app/rag/context_packer.py app/rag/service.py app/memory/context_manager.py app/api/chat.py tests/test_rag_fix.py tests/test_chat_api.py
git commit -m "fix: fail safely when rag context exceeds budget"
```

---

## Task 5：统一 RAG 与普通 Chat 的 usage、摘要和长期记忆收尾

**Files:**

- Modify: `app/services/chat_service.py`
- Modify: `tests/test_chat_api.py`

- [ ] **Step 1: 写失败测试**

使用 `ContextPolicy(recent_message_limit=2)` 和 FakeLLM 构造 RAG read 请求，验证：

1. 主模型调用一次且 `tools == []`。
2. `token_usages` 中记录 `chat`；如果本轮发生压缩，还记录 `summary` 和 `memory`。
3. 摘要成功后提炼的 MemoryEntry 在主模型成功返回后写入 MEMORY.md。
4. `ChatResponse.usage.current_turn` 包含主模型真实 usage 加上摘要/记忆维护 usage。
5. RAG 主链路仍然写入 user/assistant 原始消息，失败时不写伪造 assistant 回复。

测试使用注入的 Fake `SchemaLinkingService` 返回合成 `<schema_evidence>`，不调用 Embedding 网络。

- [ ] **Step 2: 最小实现**

将普通 `chat()` 和 `_read_response()` 的成功收尾抽成一个私有方法，至少接收：`user_id`、`session_id`、`prepared`、`result`、`route_result`。该方法统一执行：

```python
chat_usage = result.usage.model_copy(update={
    "estimated_context_tokens": prepared.estimated_context_tokens,
    "context_window": prepared.context_window,
})
session_service.record_usage(user_id, session_id, "chat", model, chat_usage)

if prepared.maintenance.compacted:
    session_service.record_usage(user_id, session_id, "summary", model, prepared.maintenance.summary_usage)
    session_service.record_usage(user_id, session_id, "memory", model, prepared.maintenance.memory_usage)
    long_term_memory.upsert(prepared.maintenance.memory_entries)

current_turn = chat_usage.add(prepared.maintenance.summary_usage).add(prepared.maintenance.memory_usage)
usage = session_service.get_session_usage(user_id, session_id, current_turn)
```

保持主聊天回答完成后再写 MEMORY.md；不要把 RAG 证据或工具原始数据写入长期记忆。空结果/歧义结果继续返回稳定追问，但不调用主模型。

- [ ] **Step 3: 验证并提交**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_chat_api.py tests/test_rag_fix.py -k "rag or usage or memory or summary" -q
git add app/services/chat_service.py tests/test_chat_api.py
git commit -m "fix: preserve rag chat usage and memory maintenance"
```

---

## Task 6：补齐完整回归、README 和最终提交

- [ ] **Step 1: 增加默认 wiring 回归测试**

至少增加一个不注入 Fake SchemaLinkingService 的测试：通过 `rag_records` 注入合成 DDL/SampleValue、通过 `rag_index_path=tmp_path` 隔离 LanceDB、mock `litellm.embedding`、注入 FakeLLM，验证真实的 `SchemaIndexer → LanceVectorStore → SchemaRetriever → ContextPacker → ChatService` 链路能够工作。

- [ ] **Step 2: 更新 README**

只补充已实现并测试验证的事实：

- 默认 catalog 只提供脱敏 Schema/DDL，不自动生成 SampleValue。
- SampleValue 必须由调用方通过 `RagSchemaRecord` 显式提供。
- SampleValue 入索引前会拒绝敏感字段和值。
- 低相关召回返回 empty；歧义返回确认提示；预算无法安全容纳证据返回 413。
- RAG 路径不执行 SQL、不读取数据行、不调用 Agent Tool Loop。

不得声称已接入真实数据库、真实业务样例数据或真实在线 Embedding 验证。

- [ ] **Step 3: 完整验证**

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m compileall -q app tests
rg -n "os\.getenv|os\.environ" app
git diff --check
git status --short
```

预期：所有测试通过、compileall 返回 0、运行时环境变量读取仍只出现在 `app/config.py`、diff 无空白错误。

- [ ] **Step 4: 提交全部改动**

```powershell
git add app tests README.md docs/v6_rag-ddl-samplevalue-fix.md requirements.txt
git commit -m "fix: harden ddl and sample value rag"
git status --short
git log -1 --oneline
```

提交后报告提交哈希、完整测试结果、未调用真实 API 的事实和任何未验证的外部行为。

## 验收标准

- 默认应用不再生成 `sample-<column>` 或任何伪造 SampleValue。
- 只有调用方显式提供的、通过敏感值过滤的 SampleValue 才进入 SampleValue 索引。
- DDL 与 SampleValue 索引仍分离；索引同步会删除已不存在的旧文档。
- 低相似度结果稳定返回 empty；Top1/Top2 过近稳定返回 ambiguous。
- 空/歧义结果不调用主聊天模型；证据无法安全打包时返回 413，不伪造回答。
- RAG fixed context 计入已有 ContextManager 预算，不静默超预算。
- RAG 成功回答和普通 chat 一样记录实际 usage、摘要 usage、memory usage，并在回答成功后安全写入 MEMORY.md。
- 原始 SQLite messages 永不删除，`user_id + session_id` 隔离保持不变。
- 所有测试不访问真实模型或真实 Embedding API。

---

## 交给另一个 Agent 的执行提示词

请在 `D:\project\python\myagent` 按 `docs\v6_rag-ddl-samplevalue-fix.md` 实际修改代码，不要只给建议。先完整阅读该计划、当前 V6 代码、已有测试和 `Agent.md`，然后严格按 Task 1 到 Task 6 执行。

必须遵守：

1. 先写失败测试并确认失败，再写最小实现；每个逻辑 Task 完成后运行对应测试。
2. 不删除已有注释，不回滚已有改动，不执行 `git reset`、`git checkout` 或删除无关文件。
3. 默认应用绝不能生成 `sample-<column>` 等伪造 SampleValue；没有外部提供的样例时只能建立 DDL 索引。
4. SampleValue 入索引前必须拒绝 password、secret、API key、token、cookie、authorization、Bearer 等敏感字段和值；测试必须证明敏感值没有进入向量索引。
5. 保留 DDL/SampleValue 双路召回、RRF、融合、`tools=[]` 和现有 API 协议；不得实现 SQL、NL2SQL、真实数据库、Workflow、SSE、MCP、Skill、向量重排模型或多 Agent。
6. 增加低相似度过滤，不能把任意 top-k 结果当作有效 Schema；empty 和 ambiguous 必须稳定可测试。
7. ContextPacker 或 ContextManager 无法在预算内安全容纳 Schema evidence 时必须抛出稳定异常并由 API 返回 413；不得继续调用主模型或伪造回答。
8. `data_operation/read` 成功回答必须复用普通 chat 的 usage、摘要和 MEMORY.md 收尾逻辑；主回答完成后才写长期记忆。
9. 所有 LiteLLM embedding 和聊天模型调用都必须 mock/Fake，不得使用真实 API Key 或真实网络。
10. 使用 PowerShell 验证：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m compileall -q app tests
rg -n "os\.getenv|os\.environ" app
git diff --check
git status --short
```

11. 修改完成后提交全部改动，不要只提交新文件：

```powershell
git add app tests README.md docs/v6_rag-ddl-samplevalue-fix.md requirements.txt
git commit -m "fix: harden ddl and sample value rag"
git status --short
git log -1 --oneline
```

最终报告必须包含：修改文件、DDL/SampleValue 数据流、敏感值过滤策略、低分/歧义/预算失败行为、RAG usage/摘要/Memory 收尾、完整测试结果、提交哈希，以及明确说明没有调用真实模型或真实 Embedding API。
