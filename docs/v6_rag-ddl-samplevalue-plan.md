# V6 DDL + SampleValue RAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax.

**Goal:** 在 data_operation/read 链路增加基于 DDL 和 SampleValue 的双路向量召回，由 SchemaLinkingService 融合候选并注入上下文，主聊天模型根据召回证据一次性回答。

**Architecture:** RAG 不是 Agent Tool Loop。data_operation/read 请求先经过 SchemaLinkingService，分别从 DDL 向量索引和 SampleValue 向量索引召回，再按表/列去重融合；融合后的候选证据由 ContextManager 纳入 Prompt，ChatAgent 使用空 Tool 列表完成一次回答。保留通用 Agent/Tool 扩展能力，但 V6 不再使用旧的 search_schema 作为 RAG Tool。

**Tech Stack:** Python 3、FastAPI、Pydantic、LiteLLM Embedding、Qwen text-embedding-v3、LanceDB、本地 JSON 索引源、SQLite 会话存储、pytest。

---

## 约束与目标链路

1. 当前 SchemaSearchTool 只做字符串包含匹配，不能作为 V6 RAG 核心；DDL 本身就是表结构，不再额外维护重复的 SchemaTable 搜索层。
2. 当前 README 的 data_operation 仍是占位响应，V6 必须改成 SchemaLinkingService 链路。
3. 当前 ChatAgent/ToolRegistry 是 V1 单 Tool Loop；V6 的 data_operation/read 路径不调用 Tool Loop，使用 tools=[] 直接完成回答。不要为了 RAG 强行保留 search_schema。
4. SQLite 继续只保存会话、消息和指标；向量索引使用独立的本地 LanceDB 目录。
5. 不实现 NL2SQL、SQL 执行、数据库连接、SSE、MCP、Skill、Workflow、向量重排模型、子 Agent 或前端 RAG 页面。
6. DDL 和 SampleValue 测试数据必须是合成数据；不得提交真实业务数据、密钥、Cookie、Token 或完整敏感样本。

README 架构图必须变为：

~~~text
Web / API Request
        ↓
ChatService → IntentRouter
                ├─ chat
                │    ↓
                │  ContextManager → ChatAgent(tools=[]) → Chat LLMAdapter
                │
                ├─ data_operation/read
                │    ↓
                │  SchemaLinkingService
                │    ├─ DDLRetriever → DDL Vector Index
                │    ├─ SampleValueRetriever → SampleValue Vector Index
                │    └─ Fusion + Context Packer
                │             ↓
                │       ContextManager → ChatAgent(tools=[]) → Chat LLMAdapter
                │
                ├─ data_operation/create/update/delete/unknown → 稳定占位响应
                └─ nl2sql → 稳定占位响应
~~~

## Qwen Embedding 配置

所有运行时环境变量读取继续只允许出现在 app/config.py。Qwen 是 V6 固定 Embedding Provider，不提供第二个向量 Provider。

建议新增：

~~~python
RAG_EMBEDDING_PROVIDER = "qwen"
RAG_EMBEDDING_MODEL = "text-embedding-v3"
RAG_EMBEDDING_LITELLM_MODEL = "dashscope/text-embedding-v3"
RAG_EMBEDDING_BASE_URL = QWEN_BASE_URL
RAG_EMBEDDING_API_KEY_ENV = QWEN_API_KEY_ENV
DEFAULT_RAG_INDEX_PATH = PROJECT_ROOT / "data" / "rag_index"

@dataclass(frozen=True)
class RagEmbeddingRuntimeConfig:
    provider: str
    api_key: str | None
    model: str
    litellm_model: str
    base_url: str | None

def get_rag_embedding_runtime_config() -> RagEmbeddingRuntimeConfig:
    return RagEmbeddingRuntimeConfig(
        provider=RAG_EMBEDDING_PROVIDER,
        api_key=os.getenv(RAG_EMBEDDING_API_KEY_ENV),
        model=RAG_EMBEDDING_MODEL,
        litellm_model=RAG_EMBEDDING_LITELLM_MODEL,
        base_url=RAG_EMBEDDING_BASE_URL,
    )
~~~

如果当前 LiteLLM 版本对 DashScope Embedding 的模型前缀有不同要求，必须以已安装版本为准，并在 config 中固定经过 mock 测试验证的模型名。业务代码不得拼接模型名或读取环境变量。

PowerShell 配置：

~~~powershell
$env:QWEN_API_KEY = "<your-qwen-key>"
~~~

缺少 QWEN_API_KEY 时，data_operation/read 的真实 RAG 请求必须返回明确 503；测试使用 FakeEmbedding，不调用网络。

## 文件职责

| 文件 | 职责 |
| --- | --- |
| app/config.py | Qwen Embedding Provider、模型、固定 Base URL、Key 环境变量名和读取函数。 |
| app/rag/models.py | RAG 文档、向量命中、融合候选、召回结果的 Pydantic 模型。 |
| app/rag/embedding.py | Embedding Protocol、Qwen Embedding Adapter、FakeEmbedding。 |
| app/rag/vector_store.py | LanceDB 两个逻辑索引的建表、upsert、删除和 top-k 搜索。 |
| app/rag/indexer.py | 将 DDL 和 SampleValue 转换为稳定索引文档，增量 upsert。 |
| app/rag/retriever.py | 两路召回、去重、RRF/加权融合、阈值判断。 |
| app/rag/context_packer.py | 在 TokenManager 预算内生成 Prompt 上下文。 |
| app/rag/service.py | SchemaLinkingService，对外协调 retrieve + pack，不调用聊天模型、不执行 SQL。 |
| app/services/chat_service.py | data_operation/read 调用 SchemaLinkingService；其他意图保持原行为。 |
| app/agent/chat_agent.py | 支持显式 tools=[] 的无 Tool 单次回答，不删除已有 Tool Loop。 |
| app/agent/tool_registry.py、app/tools/schema_search.py | 不再被 data_operation/read 作为 RAG 核心调用。 |
| README.md、requirements.txt | 更新架构、Qwen 配置和依赖。 |
| tests/test_rag_*.py、tests/test_config.py、tests/test_chat_agent.py、tests/test_chat_api.py | 完整测试。 |

## Task 1：配置和模型测试先行

**Files:** Modify tests/test_config.py; create tests/test_rag_models.py; then implement app/config.py and app/rag/models.py.

- [ ] Step 1: 写失败测试

~~~python
def test_rag_embedding_config_is_qwen_and_reads_only_qwen_key(monkeypatch):
    monkeypatch.setenv("QWEN_API_KEY", "rag-key")
    config = get_rag_embedding_runtime_config()
    assert config.provider == "qwen"
    assert config.model == "text-embedding-v3"
    assert config.litellm_model == "dashscope/text-embedding-v3"
    assert config.api_key == "rag-key"
    assert config.base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"


def test_rag_embedding_config_does_not_use_chat_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "chat-key")
    monkeypatch.delenv("QWEN_API_KEY", raising=False)
    config = get_rag_embedding_runtime_config()
    assert config.provider == "qwen"
    assert config.api_key is None
~~~

RAG 模型测试覆盖 DDL 文档、SampleValue 文档、命中结果、融合候选和空结果的严格 Pydantic 校验。

- [ ] Step 2: 确认失败

~~~powershell
.\.venv\Scripts\python.exe -m pytest tests/test_config.py::test_rag_embedding_config_is_qwen_and_reads_only_qwen_key tests/test_rag_models.py -q
~~~

- [ ] Step 3: 实现最小配置和模型并验证

~~~powershell
.\.venv\Scripts\python.exe -m pytest tests/test_config.py tests/test_rag_models.py -q
~~~

## Task 2：Qwen Embedding Adapter 和向量存储

**Files:** Create app/rag/embedding.py、app/rag/vector_store.py、tests/test_rag_embedding.py、tests/test_rag_vector_store.py；modify requirements.txt。

- [ ] Step 1: 失败测试

Mock litellm.embedding，验证 Qwen Adapter 使用 config 中的 DashScope 模型、QWEN_API_KEY 和固定 Base URL；FakeEmbedding 可替换真实 Adapter；缺 Key 抛出明确 RAGConfigurationError。VectorStore 使用临时目录，验证 ddl_documents 和 sample_value_documents 两个逻辑索引的 upsert、top-k、按 document_id 去重和删除。

- [ ] Step 2: 最小实现

定义 EmbeddingAdapter Protocol 和 QwenEmbeddingAdapter。Adapter 只负责配置检查、LiteLLM embedding 调用、向量响应校验和异常转换。VectorStore 只负责向量持久化和搜索，不读取环境变量、不调用 LiteLLM。

每条文档至少保存 document_id、source_type、namespace、table_id、table_name、column_name、text、content_hash、vector 和 metadata。

- [ ] Step 3: 安装并验证

~~~powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest tests/test_rag_embedding.py tests/test_rag_vector_store.py -q
~~~

## Task 3：DDL/SampleValue Indexer

**Files:** Create app/rag/indexer.py、tests/test_rag_indexer.py。

- [ ] Step 1: 失败测试

使用合成数据验证：一个表生成一个 DDL 文档；每个带 SampleValue 的列生成一个 SampleValue 文档；document_id 稳定；重复构建不会重复写入；内容变化更新 content_hash；没有 DDL 或 SampleValue 时跳过对应索引；SampleValue 去重、截断、脱敏；不写入完整原始行、密码、Key、Cookie、Bearer 或 Token。

建议输入模型：

~~~python
class RagSchemaRecord(BaseModel):
    namespace: str
    table_id: str
    table_name: str
    ddl: str
    columns: list[RagColumnRecord]


class RagColumnRecord(BaseModel):
    column_name: str
    data_type: str
    sample_values: list[str] = Field(default_factory=list)
~~~

- [ ] Step 2: 最小实现

SchemaIndexer 接收 EmbeddingAdapter、VectorStore 和记录列表；不得直接 import litellm 或 os。DDL 文本使用完整 DDL；SampleValue 文本使用表名、列名、类型和少量脱敏去重值。每列样本数量和字符数必须有明确上限。

- [ ] Step 3: 验证

~~~powershell
.\.venv\Scripts\python.exe -m pytest tests/test_rag_indexer.py -q
~~~

## Task 4：双路召回、融合和上下文

**Files:** Create app/rag/retriever.py、app/rag/context_packer.py、tests/test_rag_retriever.py、tests/test_rag_context_packer.py。

- [ ] Step 1: 失败测试

覆盖 DDL-only 命中、SampleValue-only 命中、同表去重、多列合并、空索引、RRF/加权稳定性、top-k/limit、低分 empty、top1/top2 过近 ambiguous、matched_by 和三类分数，以及 ContextPacker 不超过传入 Token 预算且不丢当前用户消息。

- [ ] Step 2: 最小实现

SchemaRetriever.search(query, namespace, limit) 执行：

~~~text
query embedding
  → DDL index top-k
  → SampleValue index top-k
  → table_id/column 合并
  → RRF 或明确加权融合
  → threshold/margin 判断
  → SchemaCandidate[]
~~~

优先 RRF，避免两路距离分布不可直接比较。ContextPacker 只输出有限的候选 DDL、命中列和少量 SampleValue，必须使用现有 TokenManager 估算 Token；不得把向量或全量样本塞入 Prompt。

- [ ] Step 3: 验证

~~~powershell
.\.venv\Scripts\python.exe -m pytest tests/test_rag_retriever.py tests/test_rag_context_packer.py -q
~~~

## Task 5：SchemaLinkingService 和 data_operation/read

**Files:** Create app/rag/service.py；modify app/services/chat_service.py、app/agent/chat_agent.py、app/main.py、tests/test_chat_agent.py、tests/test_chat_api.py。

- [ ] Step 1: 失败测试

使用 FakeEmbedding、FakeLLM 和临时 VectorStore 验证：data_operation/read 调用 SchemaLinkingService；执行两路召回；融合上下文进入主模型 Prompt；ChatAgent 本轮 tools=[] 调用一次并返回最终回答；模型没有 Tool call；empty 返回稳定说明而不伪造表；ambiguous 结果要求确认；create/update/delete/unknown 和 nl2sql 保持占位；缺少 QWEN_API_KEY/Embedding 服务不可用时返回 503；现有 chat 和 user_id + session_id 隔离不变。

- [ ] Step 2: 最小实现

ChatService 的 data_operation/read 链路：

~~~text
route
  → ensure session
  → SchemaLinkingService.search()
  → ContextManager.prepare(..., extra_context=packed_context)
  → ChatAgent.run(messages, tools=[])
  → 持久化 user/assistant message
~~~

SchemaLinkingService 不调用 Chat LLM。ChatAgent 增加显式无 Tool 路径，但不删除已有 Tool Loop。不要把 SchemaLinkingService 注册成 Tool，也不要在该链路调用 search_schema。RAG 上下文必须计入 Token 预算，预算不足沿用现有 413 语义。

- [ ] Step 3: 验证

~~~powershell
.\.venv\Scripts\python.exe -m pytest tests/test_chat_agent.py tests/test_chat_api.py tests/test_rag_*.py -q
~~~

## Task 6：README、依赖和完整回归

**Files:** Modify README.md、requirements.txt；按测试结果修改必要测试。

- [ ] Step 1: README 必须说明 data_operation/read 的 RAG 调用链、Qwen text-embedding-v3、QWEN_API_KEY、RAG 不执行 SQL、RAG 路径主模型一次回答且不调用 Agent Tool Loop。PowerShell 示例：

~~~powershell
$env:QWEN_API_KEY = "<your-qwen-key>"
$env:LLM_PROVIDER = "qwen"
$env:INTENT_PROVIDER = "llm"
uvicorn app.main:app --reload
~~~

不得声称实现真实数据库同步、SQL 生成、SQL 执行或在线验证。

- [ ] Step 2: 完整验证

~~~powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m compileall -q app tests
rg -n "os\.getenv|os\.environ" app
git diff --check
~~~

Expected: pytest 全部通过、compileall 为 0、运行时环境变量读取仅在 app/config.py、diff 无空白错误。

- [ ] Step 3: 检查范围

确认没有实现 NL2SQL、SQL 执行、Workflow、SSE、MCP、Skill、子 Agent、真实业务样例数据、真实模型网络测试或 RAG Tool Loop。

## 验收标准

- [ ] data_operation/read 不再返回旧占位，而是进入 SchemaLinkingService。
- [ ] DDL 和 SampleValue 两路都实际参与召回和融合。
- [ ] DDL 本身作为结构知识，不再依赖旧 search_schema。
- [ ] Qwen Embedding 的 Provider、模型、Base URL 和 API Key 环境变量名集中在 app/config.py。
- [ ] 缺 Key/Embedding 服务失败返回明确 503，不伪造成功。
- [ ] RAG 上下文受 Token 预算限制。
- [ ] 主聊天模型在 RAG 路径只做一次无 Tool 回答。
- [ ] 原有 chat、nl2sql、会话隔离、摘要和长期记忆行为回归通过。
- [ ] 所有测试不访问真实模型或真实 Embedding API。

---

## 交给另一个 Agent 的执行提示词

请在 D:\project\python\myagent 按本计划实施 V6 DDL + SampleValue RAG。不要只给建议，必须实际修改代码、测试、依赖和 README。

必须做到：

1. 先阅读本计划和当前代码，保留已有注释、已有改动和已有 V5 意图 Adapter。
2. 把 RAG 接到 README 架构图的 data_operation/read 链路：IntentRouter -> SchemaLinkingService -> DDL/SampleValue 双路向量召回 -> Fusion/ContextPacker -> ChatAgent。
3. DDL 本身就是表结构，不要再把旧 search_schema 作为 RAG 核心。data_operation/read 路径不调用 Agent Tool Loop，ChatAgent 使用 tools=[] 一次完成回答。
4. 在 app/config.py 固定 Qwen Embedding：provider=qwen、model=text-embedding-v3、LiteLLM model 使用当前版本支持的 DashScope 前缀、Base URL 使用现有 QWEN_BASE_URL，API Key 只读取 QWEN_API_KEY。业务模块不得直接读取 os.getenv/os.environ。
5. 新增 Embedding Adapter、LanceDB VectorStore、DDL/SampleValue Indexer、SchemaRetriever、Fusion、ContextPacker 和 SchemaLinkingService，保持职责分离。
6. DDL 和 SampleValue 必须分成两个逻辑索引，分别 top-k 召回后按 table_id/column_name 合并；实现稳定的 RRF 或明确加权融合、去重、empty/ambiguous、limit 和阈值。
7. 不实现 NL2SQL、SQL 执行、真实数据库同步、Workflow、SSE、MCP、Skill、子 Agent 或 RAG Tool Loop。
8. 严格 TDD：每个新行为先补失败测试并确认失败，再写最小实现。所有 LiteLLM embedding 和 Chat LLM 测试必须 mock 或 Fake，不访问真实网络，不写真实 Key，不提交真实样例数据。
9. 继承现有 ContextManager Token 预算约束；RAG 上下文必须计入 Prompt 预算，当前用户消息不可丢失。
10. 缺少 QWEN_API_KEY 或 Embedding 服务不可用时，data_operation/read 返回明确 503，不伪造候选或助手回答。
11. 运行：
~~~powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m compileall -q app tests
rg -n "os\.getenv|os\.environ" app
git diff --check
~~~
12. 最后检查 git status 和 diff，提交全部相关改动，不要只提交新增文件：
~~~powershell
git status --short
git diff --stat
git add app/config.py app/rag app/services/chat_service.py app/agent/chat_agent.py app/main.py app/schemas tests README.md requirements.txt docs/v6_rag-ddl-samplevalue-plan.md
git commit -m "feat: add ddl and sample value rag"
git status --short
git log -1 --oneline
~~~
13. 不得执行 git reset、git checkout、删除无关文件或回滚其他 Agent 改动。最终报告必须包含修改文件、DDL/SampleValue 数据流、Qwen 配置、测试完整结果、提交哈希、工作区状态，并明确说明没有调用真实模型或真实 Embedding API。
