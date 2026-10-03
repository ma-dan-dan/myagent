# 上下文、Token 与分层记忆 V3 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**目标：** 在保持当前 Agentic Chat、ToolRegistry 和多 Provider LLMAdapter 可用的前提下，实现可观测的 Token 用量、短期记忆的“历史摘要 + 最近原始消息”分层上下文，以及基于项目 `Agent.md` 和持久化 `MEMORY.md` 的长期项目记忆与按需复用。

**架构：** 原始会话消息继续完整保存到 SQLite；`ContextManager` 在每次模型调用前组装“项目规则 + 检索到的长期记忆 + 历史摘要 + 滑动窗口最近消息”，而不是把全部历史直接发送给模型。`TokenManager` 使用 LiteLLM 估算上下文 Token，并以真实 API 返回 usage 作为最终用量记录；超过预算时，`SummaryService` 对窗口外增量历史压缩并持久化。长期记忆采用 DA 的“MEMORY.md 是索引、主题文件保存细节”模式，使用轻量关键词检索，不引入向量数据库。

**技术栈：** Python 3.12、FastAPI、Pydantic v2、SQLite、LiteLLM、pytest、现有同步 `ChatAgent` 与 `SessionService`。

---

## 0. 已确认的设计决策

### 0.1 与 DA 工程的对应关系

| DA 工程实现 | 本项目 V3 对应实现 | 本次保留 / 简化点 |
| --- | --- | --- |
| `LLMBaseModel.token_count()`、`context_length()` | `TokenManager` + `data/model_context_specs.json` | 使用 LiteLLM `token_counter` 估算，未知模型走明确的保守兜底值 |
| `SessionManager` 的 SQLite 会话与 `turn_usage` | 既有 `SessionService` 扩展 `session_summaries`、`token_usages` 表 | 不引入 OpenAI Agents SDK 的表结构 |
| `_auto_compact()` / `_manual_compact()` | `SummaryService.compact_if_needed()` | 不清空原始历史；摘要和原始消息分表保存 |
| `_load_agents_md()` | `ProjectContextLoader` | 读取当前已有的 `Agent.md`，最多 200 行 |
| `memory_loader.load_memory_context()` | `LongTermMemoryStore` | 使用 `.myagent/memory/chat/MEMORY.md` 索引及主题文件；保留行数和字节上限 |
| `TokenUsage` | `TokenUsage` / `SessionUsage` Pydantic 模型 | 记录每次 Chat 与 Summary 的实际用量 |

### 0.2 当前项目的文件大小写约定

当前工作区已有 `Agent.md`。Windows 文件系统不区分 `Agent.md` 与 `AGENT.md`，因此本期必须将现有 `Agent.md` 视作需求中所说的 `AGENT.md`，不得再创建一个仅大小写不同的文件。

长期记忆不写入 `Agent.md`。`Agent.md` 是人工维护的项目规则；模型和程序只能读取它。长期可沉淀知识写入：

```text
.myagent/memory/chat/MEMORY.md
.myagent/memory/chat/topics/<topic-id>.md
```

这与 DA 的 `.da/memory/{subagent}/MEMORY.md` 思路一致，但使用当前项目自己的 `.myagent` 命名空间。

### 0.3 本期范围

1. 保留所有原始 `messages`，不丢失、不删除历史会话。
2. 新增“会话历史摘要”，摘要只覆盖已经移出滑动窗口的旧消息。
3. 每次实际 LLM 请求都记录 Provider 返回的 input/output/total Token；没有 usage 时记录为 0，并保留估算上下文 Token。
4. 用 LiteLLM 估算待发送上下文的 Token，以模型上下文上限的 80% 作为压缩触发线，预留 20% 给模型输出和工具循环。
5. `Agent.md` 每次请求都以受限长度注入系统上下文；`MEMORY.md` 只检索与当前用户消息相关的最多 3 个主题。
6. 仅在一次短期摘要成功更新后，才从摘要中提炼长期记忆候选；候选经过 Pydantic 校验和安全策略后才写入 `MEMORY.md` 主题文件。
7. 模型绝不能修改 `Agent.md`；长期记忆中不得写入 API Key、密码、Token、Cookie、完整原始对话、个人敏感信息或未经确认的模型猜测。

### 0.4 本期明确不做

1. 不实现向量数据库、Embedding、RAG 服务、跨用户共享记忆、异步任务队列、后台定时压缩或前端记忆管理页面。
2. 不改变 ToolRegistry 的工具调用协议，不新增文件写入、Shell、SQL 或任意外部工具。
3. 不实现模型自动切换、模型成本计费、Token 限流或多租户配额。
4. 不自动编辑 `Agent.md`，不将完整聊天记录复制进 `MEMORY.md`。
5. 不依赖特定 Provider 的原生 Tokenizer；实际 API usage 优先，估算仅用于上下文预算。

## 1. 目标目录与职责

| 文件 | 操作 | 责任 |
| --- | --- | --- |
| `app/memory/__init__.py` | 新建 | 记忆模块包 |
| `app/memory/models.py` | 新建 | ContextPolicy、TokenUsage、SessionUsage、SummaryRecord、MemoryEntry 等 Pydantic 模型 |
| `app/memory/token_manager.py` | 新建 | LiteLLM Token 估算、模型上下文规格读取、预算判断 |
| `app/memory/project_context.py` | 新建 | 只读加载 `Agent.md`，执行行数/字节限制 |
| `app/memory/long_term_memory.py` | 新建 | `MEMORY.md` 索引解析、关键词检索、主题文件读取、受控写入 |
| `app/memory/summary_service.py` | 新建 | 增量摘要生成、摘要持久化、长期记忆候选提炼 |
| `app/memory/context_manager.py` | 新建 | 组装最终模型上下文、触发摘要、返回上下文指标 |
| `app/storage/session_service.py` | 修改 | 增加摘要、Token 用量和按 ID 查询消息的 SQLite 方法 |
| `app/agent/adapter.py` | 修改 | 从 LiteLLM 响应提取标准 usage，附加到 `LLMResponse` |
| `app/agent/chat_agent.py` | 修改 | 聚合一轮中一个或多个模型调用的 usage，写入 `AgentResult` |
| `app/services/chat_service.py` | 修改 | 使用 ContextManager、保存用量、触发摘要后的长期记忆沉淀 |
| `app/main.py` | 修改 | 装配 TokenManager、LongTermMemoryStore、SummaryService、ContextManager |
| `app/schemas/chat.py` | 修改 | 在 `ChatResponse` 中增加只读 `usage` 字段 |
| `data/model_context_specs.json` | 新建 | 已配置模型的 context window 与输出保留 Token 规格 |
| `.myagent/memory/chat/MEMORY.md` | 新建 | 空长期记忆索引，运行时可安全增量更新 |
| `tests/test_token_manager.py` | 新建 | Token 估算、规格匹配和预算阈值测试 |
| `tests/test_summary_service.py` | 新建 | 增量摘要、失败不覆盖旧摘要、长期候选过滤测试 |
| `tests/test_long_term_memory.py` | 新建 | Agent 规则加载、索引检索、写入安全规则和边界测试 |
| `tests/test_context_manager.py` | 新建 | 最终上下文顺序、滑动窗口和压缩触发测试 |
| `tests/test_session_service.py` | 修改 | SQLite 摘要和 usage 持久化测试 |
| `tests/test_chat_agent.py` | 修改 | 多次模型调用 usage 聚合测试 |
| `tests/test_chat_api.py` | 修改 | API 返回 usage、上下文隔离、FakeLLM 回归测试 |
| `docs/v1_minimal-agentic-chat-run.md` | 修改 | 补充 Context/Memory 配置、观测与清理说明 |

## 2. 数据模型与 SQLite 设计

### 2.1 Pydantic 数据模型

在 `app/memory/models.py` 定义以下模型；所有模型使用 `ConfigDict(extra="forbid")`。

```python
class ContextPolicy(BaseModel):
    recent_message_limit: int = Field(default=12, ge=2, le=50)
    compact_ratio: float = Field(default=0.80, gt=0.50, lt=0.95)
    default_context_window: int = Field(default=16384, ge=4096)
    default_output_reserve: int = Field(default=2048, ge=256)
    summary_max_chars: int = Field(default=4000, ge=500, le=12000)
    long_term_memory_limit: int = Field(default=3, ge=0, le=10)


class TokenUsage(BaseModel):
    requests: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    estimated_context_tokens: int = Field(default=0, ge=0)
    context_window: int = Field(default=0, ge=0)


class SessionUsage(BaseModel):
    current_turn: TokenUsage
    session_total: TokenUsage


class SummaryRecord(BaseModel):
    user_id: str
    session_id: str
    summary: str = Field(min_length=1, max_length=12000)
    summarized_through_message_id: int = Field(ge=1)
    updated_at: datetime


class MemoryEntry(BaseModel):
    topic_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{0,63}$")
    title: str = Field(min_length=1, max_length=120)
    keywords: list[str] = Field(min_length=1, max_length=12)
    content: str = Field(min_length=1, max_length=4000)


class CompactionResult(BaseModel):
    attempted: bool = False
    compacted: bool = False
    summary: SummaryRecord | None = None
    summary_usage: TokenUsage = Field(default_factory=TokenUsage)
    memory_entries: list[MemoryEntry] = Field(default_factory=list)
    memory_usage: TokenUsage = Field(default_factory=TokenUsage)
```

`TokenUsage.total_tokens` 的规则：若 Provider 明确返回 total，则取该值；若只返回 input/output，则取两者之和；若没有返回 usage，则为 0。`estimated_context_tokens` 仅用于理解本次请求发送前的上下文大小，不能冒充 Provider 实际消耗。

### 2.2 SQLite 增量迁移

在 `SessionService._initialize()` 内通过 `CREATE TABLE IF NOT EXISTS` 增加，不重建、不删除既有 `sessions` 和 `messages` 表：

```sql
CREATE TABLE IF NOT EXISTS session_summaries (
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    summary TEXT NOT NULL,
    summarized_through_message_id INTEGER NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (user_id, session_id),
    FOREIGN KEY (user_id, session_id)
        REFERENCES sessions(user_id, session_id)
);

CREATE TABLE IF NOT EXISTS token_usages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    operation TEXT NOT NULL CHECK(operation IN ('chat', 'summary', 'memory')),
    model TEXT NOT NULL,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    estimated_context_tokens INTEGER NOT NULL DEFAULT 0,
    context_window INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    FOREIGN KEY (user_id, session_id)
        REFERENCES sessions(user_id, session_id)
);

CREATE INDEX IF NOT EXISTS idx_token_usages_scope
    ON token_usages(user_id, session_id, id);
```

原始 `messages` 永远不因压缩被删除。`session_summaries.summarized_through_message_id` 是短期记忆分层的边界：小于等于该 ID 的历史已被摘要覆盖；大于该 ID 的消息仍是“近期原始会话”。

## 3. 最终上下文和短期摘要的数据流

每次用户请求必须形成以下顺序；系统消息只能出现在最前部：

```text
1. 固定行为约束（现有系统提示）
2. <project_rules>：从 Agent.md 截断读取的项目规则
3. <long_term_memory>：按当前用户问题检索出的 MEMORY.md 主题
4. <conversation_summary>：此前压缩的历史摘要（若存在）
5. 最近原始消息滑动窗口：user / assistant / tool 摘要
6. 当前用户消息（此时尚未写入 SQLite，因此只添加一次）
```

`ContextManager.prepare()` 的伪接口固定为：

```python
def prepare(
    self,
    user_id: str,
    session_id: str,
    current_message: str,
) -> PreparedContext:
    ...
```

`PreparedContext` 至少包含：

```python
class PreparedContext(BaseModel):
    messages: list[dict[str, Any]]
    estimated_context_tokens: int
    context_window: int
    maintenance: CompactionResult
```

压缩判断规则：先构造候选上下文并估算 Token；当候选 Token 大于等于：

```text
(context_window - output_reserve) * compact_ratio
```

或者近期未摘要原始消息数超过 `recent_message_limit` 时，触发增量摘要。压缩目标只能是“摘要边界之后、最近窗口之前”的旧消息；当前用户消息和最近窗口消息绝不能进入本次摘要目标。

摘要成功后写入同一条 `session_summaries` 记录：

```text
新摘要 = 对（旧摘要 + 本次新移出的消息）做增量总结
新边界 = 本次被摘要的最后一条消息 ID
```

摘要失败时：保留旧摘要与旧边界，不删除原始消息；ContextManager 只发送在预算内的最近原始窗口，并在日志记录失败原因，不能返回伪造摘要。

## 4. Token 计数和用量记录规则

### 4.1 上下文预估

`TokenManager` 使用：

```python
litellm.token_counter(model=litellm_model_name, messages=messages)
```

估算上下文 Token。若 LiteLLM 因未知模型或 Tokenizer 错误抛出异常，使用稳定兜底：所有 message `content` 的 UTF-8 字节数除以 4 并向上取整，最少为 1；同时记录 warning。兜底只用于避免系统崩溃，不能显示为精确模型 Token。

模型规格从 `data/model_context_specs.json` 读取。文件格式固定为：

```json
{
  "openai": {
    "gpt-4.1-mini": {"context_window": 128000, "output_reserve": 4096}
  },
  "deepseek": {
    "deepseek-flash": {"context_window": 65536, "output_reserve": 4096}
  },
  "qwen": {
    "deepseek-v4.1-flash": {"context_window": 65536, "output_reserve": 4096}
  }
}
```

这里的 Qwen 条目仅匹配当前项目中 `QwenLLMAdapter.MODEL_ENV` 的模型名；实施 Agent 必须以当前该常量的实际值同步维护规格。找不到精确 `provider + model` 时使用 `ContextPolicy.default_context_window` 与 `default_output_reserve`，不得猜测更大的上限。

### 4.2 真实用量

`LiteLLMAdapter._to_llm_response()` 从 `response.usage` 安全提取：

```text
prompt_tokens 或 input_tokens       → input_tokens
completion_tokens 或 output_tokens  → output_tokens
total_tokens                        → total_tokens
```

并放入 `LLMResponse.usage: TokenUsage | None`。`ChatAgent.run()` 必须累加同一用户请求中的每次 `complete()` 返回 usage；工具调用导致的“模型调用 → 工具 → 模型调用”算作同一个 Chat turn 的多个 requests。

`ChatService` 将 ChatAgent 聚合 usage 写入 `token_usages(operation='chat')`。`SummaryService` 的模型调用写入 `operation='summary'`；长期记忆提炼模型调用写入 `operation='memory'`。`ChatResponse.usage` 返回：

```text
current_turn：本次聊天 AgentLoop 的聚合 usage
session_total：当前 user_id + session_id 的历史 usage 累计
```

## 5. 长期记忆规则

### 5.1 Agent.md：只读项目规则

`ProjectContextLoader.load_rules()`：

1. 仅从项目根目录读取既有 `Agent.md`。
2. 最多读取 200 行、25 KB；超出时追加明确的截断提示。
3. 读取失败或不存在时返回空字符串，不中断 Chat。
4. 将内容包裹为 `<project_rules>...</project_rules>` 系统消息。
5. 永远不向该文件写入任何内容。

### 5.2 MEMORY.md：索引与主题文件

`MEMORY.md` 只能保存一行索引条目：

```markdown
- [LLM Adapter 配置](topics/llm-adapter.md) | keywords: llm, provider, qwen | 模型接入层的稳定配置决策
```

主题文件保存有 frontmatter 和正文：

```markdown
---
topic_id: llm-adapter
title: LLM Adapter 配置
keywords: [llm, provider, qwen]
updated_at: 2026-09-30T00:00:00+00:00
---

模型 Provider 由 LLM_PROVIDER 选择；API Key 从系统环境变量读取；模型名由对应 Provider 类中的 MODEL_ENV 常量配置。
```

`LongTermMemoryStore.retrieve(query)` 的规则：

1. 将 query 规范化为小写关键词，按中英文空白、标点切分；长度小于 2 的 token 丢弃。
2. 在索引的 title、keywords、hook 中做关键词匹配；按“关键词命中数、标题命中”排序。
3. 最多读取 `long_term_memory_limit=3` 个主题文件，每个主题最多 2,000 字符，总内容最多 6,000 字符。
4. 无匹配时返回空列表；不能把完整 `MEMORY.md` 与所有主题文件无差别塞入模型上下文。

### 5.3 连续沉淀，但受控写入

一次短期摘要成功后，`SummaryService` 使用同一个 LLMAdapter 发起一次无工具的“长期记忆提炼”请求。提示词要求只输出 JSON 数组，元素对应 `MemoryEntry`：

```json
[
  {
    "topic_id": "llm-adapter",
    "title": "LLM Adapter 配置",
    "keywords": ["llm", "provider", "qwen"],
    "content": "模型 Provider 由 LLM_PROVIDER 选择。"
  }
]
```

处理规则必须固定：

1. 用 `json.loads()` 解析；解析失败时不写入长期记忆，记录一次 `memory` usage 后继续正常聊天。
2. 用 `list[MemoryEntry]` 的 Pydantic TypeAdapter 校验；校验失败时不写入。
3. `MemorySafetyPolicy` 拒绝包含下列词或模式的内容：`api_key`、`password`、`secret`、`token`、`cookie`、`authorization`、`sk-`、`Bearer `，以及长度超过模型定义上限的内容。
4. 只接受“用户明确的长期偏好、稳定项目规则、已验证架构决策、可跨会话复用的项目知识”；拒绝一次性任务、原始聊天逐字稿、工具原始输出、未经验证猜测。
5. 同一 `topic_id` 已存在时，覆盖对应主题文件并更新索引 hook；不存在时创建主题文件并追加索引行。
6. 写入使用 `Path.resolve()` 验证目标始终位于 `.myagent/memory/chat/` 内，禁止路径穿越。
7. 长期记忆提炼失败不影响当前聊天响应，也不回滚已成功保存的短期摘要。

## 6. 实施任务

### Task 1：先建立模型与 Session 存储失败测试

**文件：**

- 新建：`app/memory/__init__.py`
- 新建：`app/memory/models.py`
- 修改：`app/storage/session_service.py`
- 修改：`tests/test_session_service.py`

- [ ] **Step 1：为摘要边界和 usage 聚合写失败测试。**

在 `tests/test_session_service.py` 建立临时 SQLite 数据库，追加 4 条消息后保存摘要，断言摘要边界和原始消息都能读取：

```python
service.save_summary("alice", "session-a", "旧消息摘要", summarized_through_message_id=2)

summary = service.get_summary("alice", "session-a")
messages = service.get_messages_after_id("alice", "session-a", 2)

assert summary.summary == "旧消息摘要"
assert summary.summarized_through_message_id == 2
assert [item.content for _, item in messages] == ["第三条", "第四条"]
```

再记录两条 usage，断言同一用户与 Session 的累加值正确、另一用户同名 Session 看不到用量：

```python
assert service.get_session_usage("alice", "session-a").total_tokens == 30
assert service.get_session_usage("bob", "session-a").total_tokens == 0
```

- [ ] **Step 2：运行失败测试。**

运行：

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_session_service.py -q
```

预期：因 `save_summary`、`get_summary`、`get_messages_after_id`、`record_usage`、`get_session_usage` 和记忆模型不存在而失败。

- [ ] **Step 3：实现 Pydantic 模型和 SQLite 增量迁移。**

实现第 2 节列出的模型和表。`SessionService` 必须新增如下公开方法：

```python
def get_messages_with_ids(self, user_id: str, session_id: str) -> list[tuple[int, StoredMessage]]: ...
def get_messages_after_id(self, user_id: str, session_id: str, message_id: int) -> list[tuple[int, StoredMessage]]: ...
def get_summary(self, user_id: str, session_id: str) -> SummaryRecord | None: ...
def save_summary(self, user_id: str, session_id: str, summary: str, summarized_through_message_id: int) -> None: ...
def record_usage(self, user_id: str, session_id: str, operation: str, model: str, usage: TokenUsage) -> None: ...
def get_session_usage(self, user_id: str, session_id: str) -> TokenUsage: ...
```

所有方法必须复用现有 `_validate_identifier`，所有 SQL 必须始终以 `user_id + session_id` 过滤。

- [ ] **Step 4：运行测试确认通过。**

运行：

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_session_service.py -q
```

预期：通过，且既有 Session 隔离断言仍通过。

### Task 2：实现 Token 规格、估算和真实 usage 映射

**文件：**

- 新建：`data/model_context_specs.json`
- 新建：`app/memory/token_manager.py`
- 修改：`app/agent/adapter.py`
- 修改：`app/agent/chat_agent.py`
- 新建：`tests/test_token_manager.py`
- 修改：`tests/test_llm_adapter.py`
- 修改：`tests/test_chat_agent.py`

- [ ] **Step 1：写 TokenManager 的失败测试。**

使用注入的假计数函数而不调用实际 Tokenizer：

```python
counter = TokenManager(
    specs_path=specs_path,
    provider="qwen",
    model="deepseek-v4.1-flash",
    count_messages=lambda messages: 120,
)

assert counter.context_window == 65536
assert counter.output_reserve == 4096
assert counter.should_compact(50000) is True
assert counter.should_compact(1000) is False
```

再测试未知模型使用 `ContextPolicy` 的保守默认值，以及计数函数抛出异常时的 UTF-8 字节兜底值。

- [ ] **Step 2：写 LiteLLM usage 映射和 Agent 聚合的失败测试。**

构造 LiteLLM 假响应：

```python
response.usage = SimpleNamespace(
    prompt_tokens=11,
    completion_tokens=7,
    total_tokens=18,
)
```

断言 Adapter 返回的 `LLMResponse.usage` 为 `TokenUsage(input_tokens=11, output_tokens=7, total_tokens=18, requests=1)`。再让 FakeLLM 依次返回 Tool Call（usage 10）和最终消息（usage 20），断言 `AgentResult.usage.total_tokens == 30`、`requests == 2`。

- [ ] **Step 3：运行失败测试。**

运行：

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_token_manager.py tests\test_llm_adapter.py tests\test_chat_agent.py -q
```

预期：因 TokenManager、usage 字段和 Agent 聚合未实现而失败。

- [ ] **Step 4：实现 TokenManager。**

`TokenManager` 必须：

```python
class TokenManager:
    def estimate_messages(self, messages: list[dict[str, Any]]) -> int: ...
    def should_compact(self, estimated_tokens: int) -> bool: ...
    @property
    def context_window(self) -> int: ...
    @property
    def output_reserve(self) -> int: ...
```

生产计数函数调用 `litellm.token_counter(model=litellm_model_name, messages=messages)`；单元测试通过构造参数注入计数函数。不得在单元测试访问模型网络。

- [ ] **Step 5：扩展 Adapter 和 ChatAgent，但不改变 `complete()` 签名。**

`LLMResponse` 新增：

```python
usage: TokenUsage | None = None
```

`AgentResult` 新增：

```python
usage: TokenUsage = Field(default_factory=TokenUsage)
```

`LiteLLMAdapter._to_llm_response()` 负责解析 Provider usage；`ChatAgent.run()` 在每次 `self.llm.complete(...)` 后累加 usage。FakeLLM 可以省略 usage，表示 0；不得为了测试而要求 FakeLLM 调用 LiteLLM。

- [ ] **Step 6：运行 Token 和 Agent 测试。**

运行：

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_token_manager.py tests\test_llm_adapter.py tests\test_chat_agent.py -q
```

预期：全部通过。

### Task 3：实现只读项目规则和长期记忆索引检索

**文件：**

- 新建：`app/memory/project_context.py`
- 新建：`app/memory/long_term_memory.py`
- 新建：`.myagent/memory/chat/MEMORY.md`
- 新建：`tests/test_long_term_memory.py`

- [ ] **Step 1：写 Agent.md 读取和截断失败测试。**

在临时 workspace 创建 205 行的 `Agent.md`，断言加载内容只含前 200 行并包含截断提示；缺失文件返回空字符串；加载过程不写文件。

- [ ] **Step 2：写长期记忆检索和安全写入失败测试。**

创建一个索引和两个主题文件，断言：

```python
results = store.retrieve("Qwen 的模型接入配置是什么")
assert [item.topic_id for item in results] == ["llm-adapter"]
```

再断言以下内容拒绝写入：

```python
MemoryEntry(
    topic_id="secret",
    title="Key",
    keywords=["key"],
    content="api_key=sk-example",
)
```

并断言 `../outside.md` 的 topic ID 或路径不能写到记忆根目录之外。

- [ ] **Step 3：运行失败测试。**

运行：

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_long_term_memory.py -q
```

预期：因 `ProjectContextLoader`、`LongTermMemoryStore` 与安全策略不存在而失败。

- [ ] **Step 4：实现 ProjectContextLoader。**

实现：

```python
class ProjectContextLoader:
    def load_rules(self) -> str: ...
```

构造参数接收 `workspace_root`，仅检查 `workspace_root / "Agent.md"`。读取上限为 200 行和 25 KB；读取异常返回空字符串。不得扫描其他目录，不得创建或编辑 `Agent.md`。

- [ ] **Step 5：实现 LongTermMemoryStore。**

实现：

```python
class LongTermMemoryStore:
    def retrieve(self, query: str) -> list[MemoryEntry]: ...
    def upsert(self, entries: list[MemoryEntry]) -> list[MemoryEntry]: ...
```

`retrieve()` 必须严格遵守第 5.2 节的关键词、条数和字符限制。`upsert()` 必须执行第 5.3 节的安全策略与 `Path.resolve()` 根目录验证，采用临时文件后原子 `replace()` 写入主题文件，最后更新 `MEMORY.md` 索引。任何单个条目失败不能破坏已有索引或已有主题文件。

- [ ] **Step 6：运行长期记忆测试。**

运行：

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_long_term_memory.py -q
```

预期：全部通过，且测试中不存在网络请求。

### Task 4：实现增量摘要和上下文组装

**文件：**

- 新建：`app/memory/summary_service.py`
- 新建：`app/memory/context_manager.py`
- 新建：`tests/test_summary_service.py`
- 新建：`tests/test_context_manager.py`

- [ ] **Step 1：写增量摘要失败测试。**

使用一个 FakeLLM，确保摘要调用没有工具：

```python
summary_llm.calls[0]["tools"] == []
```

给定旧摘要 `"用户关注产量"` 和两条已移出窗口的原始消息，FakeLLM 返回 `"用户关注产量，并要求按季度查看。"`；断言新摘要被保存且 `summarized_through_message_id` 前进。再让 FakeLLM 抛出 `LLMServiceUnavailable`，断言旧摘要和旧边界保持不变。

- [ ] **Step 2：写上下文顺序和滑动窗口失败测试。**

准备：项目规则、一个长期记忆主题、旧摘要、6 条原始消息，并将最近窗口设置为 2。断言最终 `PreparedContext.messages` 中的顺序为：

```python
["system:fixed", "system:project_rules", "system:long_term_memory", "system:conversation_summary", "最近两条原始消息", "当前用户消息"]
```

并断言旧消息既不会重复进入原始窗口，也不会在没有摘要覆盖时被悄悄丢弃。

- [ ] **Step 3：运行失败测试。**

运行：

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_summary_service.py tests\test_context_manager.py -q
```

预期：因 SummaryService、ContextManager、PreparedContext 不存在而失败。

- [ ] **Step 4：实现 SummaryService。**

实现接口：

```python
class SummaryService:
    def compact_if_needed(
        self,
        user_id: str,
        session_id: str,
        current_message: str,
    ) -> CompactionResult: ...

    def extract_long_term_memory(self, summary: str) -> tuple[list[MemoryEntry], TokenUsage]: ...
```

摘要请求使用 `llm.complete(summary_messages, tools=[])`，提示词必须明确要求：保留用户目标、已确认事实、约束、未解决问题和工具得到的重要结论；不能编造事实；输出纯文本摘要，最大 `summary_max_chars` 字符。摘要输入必须包含旧摘要和“本次新移出窗口的消息”，而不是整个会话。

记忆提炼请求同样使用无工具 `complete()`；若返回不是普通 message、JSON 解析失败或 Pydantic 校验失败，返回空列表且不抛出到用户聊天路径。无论摘要还是记忆提炼，都要返回本次调用的 usage，供 ChatService 记录。

- [ ] **Step 5：实现 ContextManager。**

`ContextManager.prepare()` 的固定动作：

1. 调用 `SummaryService.compact_if_needed()`。
2. 读取更新后的摘要。
3. 获取摘要边界之后的所有消息，取最后 `recent_message_limit` 条；若未摘要消息仍超出窗口而摘要失败，只取最近窗口并记录 warning，不更新摘要边界。
4. 加载 `Agent.md` 规则。
5. 根据 `current_message` 检索长期记忆。
6. 依照第 3 节的固定顺序生成消息。
7. 使用 TokenManager 估算最终消息；若仍超过 `context_window - output_reserve`，优先截断长期记忆，再截断项目规则，绝不截断当前用户消息；记录 warning 和最终估算值。

- [ ] **Step 6：运行摘要与上下文测试。**

运行：

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_summary_service.py tests\test_context_manager.py -q
```

预期：全部通过。

### Task 5：接入 ChatService、API 和 usage 观测

**文件：**

- 修改：`app/services/chat_service.py`
- 修改：`app/schemas/chat.py`
- 修改：`app/main.py`
- 修改：`tests/test_chat_api.py`

- [ ] **Step 1：写 API 失败测试。**

使用注入的 FakeLLM 和临时数据库，发送两轮请求。断言：

```python
payload = response.json()
assert payload["usage"]["current_turn"]["requests"] == 1
assert payload["usage"]["session_total"]["requests"] == 2
assert "conversation_summary" in json.dumps(fake.calls[1], ensure_ascii=False)
```

另写测试断言不同 `user_id` 即使使用相同 `session_id`，不能读取对方摘要、usage 或长期会话上下文。

- [ ] **Step 2：运行失败测试。**

运行：

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_chat_api.py -q
```

预期：因 `usage` 响应字段与 ContextManager 装配不存在而失败。

- [ ] **Step 3：修改 ChatService。**

`ChatService.chat()` 的顺序必须是：

```text
ensure/create session
→ ContextManager.prepare(...)
→ append 当前 user 原始消息
→ ChatAgent.run(prepared.messages)
→ append tool events 与 assistant 原始消息
→ record chat usage
→ 根据 prepared.maintenance 记录 summary/memory usage 并 upsert 安全长期记忆
→ 查询 session usage
→ 返回 ChatResponse
```

移除当前 `ChatService` 中自行拼装完整历史的循环；该职责只能由 ContextManager 承担。Tool message 仍保存到 SQLite，摘要输入可以使用其简短 summary，但不能把工具原始大结果写入长期记忆。

- [ ] **Step 4：扩展 API 模型。**

在 `ChatResponse` 增加：

```python
usage: SessionUsage
```

此字段必须始终存在，即使 FakeLLM 或 Provider 没有返回 usage；此时 Token 数为 0，但 `estimated_context_tokens` 和 `context_window` 仍应反映 ContextManager 的估算。

- [ ] **Step 5：在 main.py 装配依赖。**

`create_app()` 必须从现有的 LLMAdapter 读取 Provider、模型名和 LiteLLM 模型名，并创建共享的：

```python
TokenManager
ProjectContextLoader
LongTermMemoryStore
SummaryService
ContextManager
```

测试通过 `create_app(..., llm_adapter=fake)` 注入 FakeLLM 时，需要允许额外传入确定性 TokenManager 或 ContextPolicy，不能要求 FakeLLM 暴露 LiteLLM 私有属性。

- [ ] **Step 6：运行 API 回归测试。**

运行：

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_chat_api.py tests\test_chat_agent.py -q
```

预期：全部通过；现有用户/Session 隔离、工具调用和 503 错误映射语义不能改变。

### Task 6：补充运行文档并全量验证

**文件：**

- 修改：`docs/v1_minimal-agentic-chat-run.md`
- 修改：`requirements.txt`（仅在 LiteLLM 的当前依赖不包含所需 Tokenizer 时；否则不改）

- [ ] **Step 1：更新运行说明。**

说明以下内容：

1. `Agent.md` 是只读项目规则；不能放 API Key。
2. `.myagent/memory/chat/MEMORY.md` 是长期记忆索引，主题文件位于 `topics/`；删除该目录只会清除长期记忆，不会清除 SQLite 会话。
3. `data/model_context_specs.json` 的 provider/model 规格需要与 `app/agent/llm_providers.py` 的 Provider 和模型常量保持一致。
4. `POST /api/v1/chat` 的响应新增 `usage.current_turn` 和 `usage.session_total`。
5. Token 估算用于压缩阈值，Provider 返回 usage 才是实际用量；未知模型会采用保守兜底。

- [ ] **Step 2：执行全量验证。**

运行：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m compileall -q app
rg -n "Agent\.md|MEMORY\.md|session_summaries|token_usages|conversation_summary|long_term_memory" app tests docs
```

预期：所有 pytest 通过，Python 编译无输出，搜索结果只出现在本期设计的模块、测试和运行文档中。

- [ ] **Step 3：人工验证一次真实会话。**

在不输出 API Key 的前提下：连续发送足够多的消息触发低阈值测试配置，确认响应中有 usage、SQLite 中有 summary/usage、下一轮模型输入同时含 `<conversation_summary>` 和最近原始消息；随后新建 Session，确认与问题相关的长期主题可被检索，但 `Agent.md` 未被修改。

## 7. 验收标准

1. 原始会话消息保留在 SQLite，摘要只是一份单独的增量记忆，不删除原始数据。
2. 最终模型上下文严格按“项目规则 → 长期记忆 → 会话摘要 → 近期原始消息 → 当前问题”排序。
3. 不会将历史摘要覆盖范围内的旧消息重复发送给模型。
4. 上下文超过预算时优先增量摘要；摘要失败时不覆盖旧摘要，不执行破坏性删除。
5. 每次真实模型调用的 usage 能被抽取、每轮 AgentLoop 能被聚合、每 Session 能被累计；无 usage 的 FakeLLM 仍可工作。
6. `Agent.md` 只读且受 200 行/25 KB 上限保护。
7. 长期记忆按查询关键词最多取 3 个主题，不会无差别注入全部历史。
8. 长期记忆提炼只能在摘要成功后触发，且不会写入密钥、Token、Cookie、原始对话或 `Agent.md`。
9. 所有 SQLite 查询继续按 `user_id + session_id` 隔离；跨用户不能读到对方的摘要、usage 或会话消息。
10. 全量 pytest 和 `compileall` 通过；测试不调用真实模型网络。

## 8. 交给实施 Agent 的提示词

将以下整段原样发送给实施 Agent：

```text
请在 D:\project\python\myagent 按 docs\v3_context-token-memory-management.md 实施“上下文、Token 与分层记忆 V3”。本次只实现该计划，不实现向量数据库、MCP、Skill、流式、多 Agent、模型路由或前端记忆页面。

开始前请完整阅读：

- docs\v3_context-token-memory-management.md
- app\storage\session_service.py
- app\services\chat_service.py
- app\agent\chat_agent.py
- app\agent\adapter.py
- app\agent\llm_factory.py
- app\agent\llm_providers.py
- app\schemas\chat.py
- tests\test_session_service.py
- tests\test_chat_agent.py
- tests\test_chat_api.py
- D:\project\qiuzhao\DA\da\models\session_manager.py
- D:\project\qiuzhao\DA\da\agent\node\agentic_node.py
- D:\project\qiuzhao\DA\da\utils\memory_loader.py
- D:\project\qiuzhao\DA\da\schemas\token_usage.py

实现原则：

1. 保留现有 ChatAgent 的 LLMAdapter.complete(messages, tools) 调用契约、ToolRegistry、SQLite 原始消息、API 路由和多 Provider LLMAdapter；不删除已有注释或无关逻辑。
2. 按 TDD：每个新行为先补失败测试，确认失败后再写最小实现。所有 LiteLLM 调用必须 mock；不得使用真实 API Key 或真实模型网络。
3. 原始 messages 永远不删除。使用 session_summaries 保存“旧摘要 + 摘要边界”，使用滑动窗口保留近期原始消息。
4. 估算 Token 用 LiteLLM token_counter，真实用量从 Provider response.usage 抽取；估算和实际用量必须在数据模型中明确区分。
5. 当前已有 Agent.md；将其当作需求中的 AGENT.md，只读加载，绝不创建大小写不同的重复文件，也绝不让模型写入。
6. 长期记忆采用 .myagent\memory\chat\MEMORY.md 索引 + topics\ 主题文件。只允许安全、稳定、可复用的项目知识进入；拒绝密钥、Token、Cookie、密码、原始聊天记录与未经确认的猜测。
7. 任何摘要或长期记忆提炼失败都不得影响当前用户聊天、不得覆盖旧摘要、不得损坏 MEMORY.md。
8. 当前目录不是 Git 仓库，不执行 git commit、reset、checkout 或删除无关文件。
9. 不新增代码注释；若确实无法避免，必须遵循项目已有语言语法且不得删除已有注释。

完成后报告：实际修改文件、短期/长期记忆数据流、Token 估算与真实 usage 的区别、测试命令和完整结果、以及手动验证步骤。不要声称已调用真实模型，除非确实获得并使用了用户明确提供的凭据。
```
