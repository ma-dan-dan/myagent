# 最小 Agentic Chat V1 设计文档

## 1. 目标与范围

本版本先暂停固定 Workflow NL2SQL，交付一个可以从简单网页发消息、由后端维持多轮上下文、并能按需调用一个只读 Tool 的开放式 Agentic Chat。

用户的可见效果是：

1. 用户在网页输入问题并发送。
2. 后端返回助手回复。
3. 同一个 `user_id + session_id` 的后续消息会带上历史上下文。
4. 当问题需要了解数据表结构时，模型可以调用 `search_schema`，再根据 Tool 返回结果回复。
5. 当信息不足时，模型以普通助手消息追问；用户下一次发送补充信息后，系统通过 Session 继续对话。

最小例子：用户问“产量相关的表有哪些？”，模型调用 `search_schema`；Tool 返回表名、表注释、字段名、字段类型和字段注释；模型再用自然语言解释哪些表可能有用。

## 2. 明确不做

以下能力不是 V1 范围，禁止为了“完整”而提前加入：

- 固定 NL2SQL Workflow、SQL 生成、DuckDB 查询和 Reflection。
- SSE、断线重连、后台任务、流式 token 输出。
- MCP、Skill、子 Agent、工具权限审批、文件系统或 Shell Tool。
- Sample Value RAG、真实业务数据样例、向量库、LanceDB、Reranker。
- 登录鉴权、多租户服务、模型路由和多个模型提供商。
- 复杂的 `ask_user` 暂停/恢复协议。

V1 的目的不是立即解决数据分析，而是先把“Session + 模型 + Tool + 模型”的 Agent Loop 跑清楚、测清楚、讲清楚。

## 3. 与原项目的对应关系

原项目 `D:\project\qiuzhao\DA` 的开放聊天链路包含大量生产级能力。V1 借鉴其分层方式，不能整包照搬其数据分析业务依赖。

| V1 职责 | 原项目参考 | V1 取舍 |
| --- | --- | --- |
| HTTP 聊天入口 | `da/api/routes/chat_routes.py` | 保留单个普通 JSON 接口，不做 SSE。 |
| 聊天业务编排 | `da/api/services/chat_service.py` | 保留“路由薄、服务层编排”的职责划分。 |
| Agent 循环与 Tool 装配 | `da/agent/node/chat_agentic_node.py` | 保留模型可调 Tool 的思想，只放一个本地 Tool。 |
| Session 生命周期 | `da/models/session_manager.py` | 保留 SQLite 和按 `user_id` 隔离；不复制分支、压缩和回退。 |
| 模型 Tool Loop | `da/models/openai_compatible.py`、`da/models/base.py` | 参考 OpenAI Agents SDK 的 `Runner` 驱动方式，不复制多厂商适配层。 |
| 人机交互 | `da/tools/func_tool/ask_user_tools.py` | V1 改用“本轮文字追问、下一轮继续”的简单模式。 |

原项目中 `ChatAgenticNode` 将 Tools、MCP Servers、Session 和最大轮数交给模型执行层；模型在 Tool 返回后继续推理。原项目固定 NL2SQL 的 `ReflectNode` 则是另一条 Workflow 链路的 SQL 质量检查员，V1 不实现它。

## 4. V1 总体架构

```text
简单网页
  │ POST /api/v1/chat
  ▼
FastAPI Chat Route
  ▼
ChatService
  ├─ SessionService：读取、创建和保存 SQLite 会话
  └─ ChatAgent：构造模型输入并运行 Agent Loop
       ├─ LLM Adapter：唯一的模型访问边界
       └─ ToolRegistry：只注册 search_schema
            └─ SchemaCatalog：读取脱敏的结构元数据目录
  ▼
ChatResponse
  └─ session_id、assistant_message、可选 tool_trace
```

职责边界必须保持清楚：

- Route 只处理 HTTP 与 Pydantic 请求/响应模型，不能写 Agent 逻辑。
- ChatService 负责一次会话请求的编排，不能读取具体 Schema 文件。
- ChatAgent 负责模型与 Tool 的循环，不能拼接 SQL 或直接操作 SQLite 文件。
- `search_schema` 只检索元数据，不能读取真实表数据，也不能返回样例行。
- SessionService 只持久化与读取历史，不能决定模型下一步做什么。

## 5. 请求、响应与状态

### 5.1 请求

`POST /api/v1/chat` 接收以下语义字段：

- `user_id`：调用方标识。V1 没有登录系统，但每次请求必须显式提供它。
- `session_id`：可选。首次对话为空；服务端创建后由响应返回，后续请求带回。
- `message`：当前用户消息，不能为空，并设置合理长度上限。

### 5.2 响应

响应至少包含：

- `session_id`：本轮使用或新创建的会话标识。
- `message`：助手最终自然语言回复。
- `tool_events`：本轮 Tool 调用的简短、可展示轨迹；无调用时为空列表。

不能把底层模型原始响应、完整历史或异常堆栈直接返回给浏览器。

### 5.3 Session

Session 的身份是 `user_id + session_id`，两者缺一不可。推荐每个用户拥有独立 SQLite 文件或在同一 SQLite 数据库中用 `user_id` 强制过滤；无论选哪种，外层 API 都必须确保用户 A 无法通过猜测 `session_id` 读取用户 B 的历史。

每轮至少保存：用户消息、助手消息、Tool 调用名称与参数摘要、Tool 返回摘要、创建时间。模型的下一轮输入只读取最近有限轮历史，避免上下文无限增长。

## 6. Tool 设计

### 6.1 唯一 Tool：search_schema

输入是查询关键词和受服务端限制的最大返回条数。输出只能包含：

- 表名；
- 表注释 / 业务含义；
- 字段名；
- 字段类型；
- 字段注释。

检索目录使用项目内预置的脱敏 Schema 元数据文件。V1 可以采用大小写不敏感的关键词匹配；后续可在不改变 Tool 对外契约的前提下替换成 DDL 检索、SQLite FTS 或 LanceDB 向量检索。

空结果是正常结果，不是异常。Tool 应返回“没有匹配项”的结构化状态；模型据此向用户说明当前元数据中没有相关表，或请用户换一种业务名称。

### 6.2 Agent 决策规则

- 一般问答：模型直接回复，不调用 Tool。
- 询问表、字段、可用数据结构：模型调用 `search_schema`。
- 用户问题缺少业务条件，例如“查产量”但没有时间范围和对象：模型直接追问，不调用 Tool。
- Tool 返回空结果或错误：模型使用可解释的自然语言回复，不进行无限重试。
- 每轮请求设置最大 Agent 轮数和最大 Tool 调用次数，V1 建议均不超过 2；超限时返回清楚的失败说明。

“追问用户”在 V1 是普通助手消息，不是 Tool，也不会让 HTTP 请求保持阻塞。用户下一次提交补充信息时，由 Session 恢复上下文。

## 7. Agent Loop 数据流

```text
1. 浏览器发送 user_id、可选 session_id、message
2. Route 完成格式校验并交给 ChatService
3. ChatService 创建或定位同一用户下的 Session
4. SessionService 读取有限历史，加上当前用户消息
5. ChatAgent 把系统指令、历史和 search_schema 的工具描述交给模型
6. 模型选择：直接回复，或请求调用 search_schema
7. 后端验证 Tool 参数、执行元数据检索、记录调用事件
8. Tool 结果回到模型；模型生成最终文字回复，或在上限内继续调用
9. SessionService 保存本轮消息和 Tool 轨迹
10. ChatService 返回 session_id、回复和简短轨迹
```

Tool 结果回到模型不只是为了“校验正确性”。后端负责 Tool 参数、允许范围、超时和返回格式；模型负责判断结果是否足够回答用户、如何解释结果，以及是否仍需行动。

## 8. 推荐目录与文件职责

实现时可采用以下小型结构；若实施者已有更清晰的等价结构，可以调整目录，但不得混合职责。

```text
app/
  main.py                    # FastAPI 应用组装与路由注册
  config.py                  # 环境配置和路径配置
  api/chat.py                # POST /api/v1/chat 路由
  schemas/chat.py            # 请求、响应、Tool 事件的 Pydantic 模型
  services/chat_service.py   # Session、Agent、响应编排
  agent/chat_agent.py         # 系统指令、Agent Loop、最大轮数控制
  agent/tool_registry.py      # 仅注册允许的 Tool
  tools/schema_search.py      # search_schema 的输入校验与调用入口
  storage/session_service.py  # SQLite 会话读写与 user_id 隔离
  storage/schema_catalog.py   # Schema 元数据文件加载和关键词检索
data/
  schema_catalog.json         # 脱敏的演示表/字段元数据
web/
  index.html                  # 无框架的最小消息发送与回复展示页
tests/
  test_schema_search.py
  test_session_service.py
  test_chat_agent.py
  test_chat_api.py
```

## 9. 技术选择

- Python、FastAPI、Pydantic、SQLite。
- 模型 Tool Loop 优先采用 OpenAI Agents Python SDK。原项目依赖 `openai-agents`，并在模型适配层调用 SDK 的 `Runner`。V1 只保留一个模型提供商和一个模型配置。
- Session 优先使用 SDK 支持的 SQLite Session；若当前 SDK 版本的接口不稳定或不满足 `user_id` 隔离要求，将 SDK Session 封装在 `SessionService` 后面，不能让 Route 或 Tool 依赖其具体类型。
- 测试时必须可注入假的 LLM Adapter，绝不能要求测试真实访问模型 API。
- 简单网页只使用原生 HTML、CSS、JavaScript；不引入前端框架。

## 10. 安全、失败与可观测性

- 只暴露显式注册的 `search_schema`；禁止把文件、Shell、数据库执行等能力注册给模型。
- 校验 `user_id` 与 `session_id` 的字符集，防止路径穿越或跨用户读取。
- Tool 参数采用 Pydantic 校验，返回值也使用固定模型。
- 模型、Tool、SQLite 错误都转换为稳定的应用错误，不暴露密钥、路径和调用栈。
- 记录 request/session 标识、Agent 轮数、Tool 名称、耗时、成功或失败状态；日志不记录完整敏感数据。
- 前端只展示安全的文字和经服务端整理的 Tool 摘要。

## 11. 验收标准

V1 完成必须同时满足：

1. 浏览器可以发送消息并收到 JSON 回复。
2. 首轮未传 `session_id` 时，服务端创建并返回它；后续同用户携带该 ID 时能看见上下文。
3. 不同 `user_id` 即使使用相同 `session_id` 也不能共享历史。
4. 模型对 Schema 相关问题能够调用 `search_schema`，并基于结果回复。
5. `search_schema` 空结果、Tool 异常、模型异常和无效输入都有可解释响应。
6. 单次请求不会无限调用 Tool。
7. 自动化测试不调用真实模型 API，覆盖 Session 隔离、Tool 检索、Tool 空结果、Agent Tool 分支与 API 主链路。
8. 没有加入 SQL、Workflow、MCP、Skill、SSE 或真实样例数据。

## 12. 后续演进顺序

V1 稳定后再逐项增加：

1. SSE，将当前一次性回复改为过程事件流。
2. 受控的人机交互事件，而不是仅通过下一轮消息追问。
3. Schema 元数据检索升级为 DDL + 向量检索。
4. 将固定 NL2SQL Workflow 封装为一个新的只读 Tool。
5. 再评估是否接入 MCP 和 Skill。

这个顺序保证每一层都有稳定的下层依赖：先验证 Agent Loop，再增加数据分析能力，而不是一开始把所有复杂系统耦合在一起。

## 13. 交给实现 Agent 的提示词

将下面整段提示词发送给实施 Agent，并同时提供本设计文档与 `Agent.md`：

> 你要在 `D:\project\python\myagent` 实现“最小 Agentic Chat V1”。开始前完整阅读 `Agent.md` 和 `docs/minimal-agentic-chat-design.md`，并以它们为最高项目要求。再只读参考原项目 `D:\project\qiuzhao\DA` 中的 `da/api/routes/chat_routes.py`、`da/api/services/chat_service.py`、`da/api/services/chat_task_manager.py`、`da/agent/node/chat_agentic_node.py`、`da/agent/node/agentic_node.py`、`da/models/session_manager.py`、`da/models/openai_compatible.py`。参考其职责划分与 Agent SDK 的 Tool Loop，不要复制 DA 的 SQL、MCP、Skill、权限、SSE 或多模型实现。
>
> 交付范围严格限定为：一个原生 HTML 页面；一个 FastAPI 普通 JSON 聊天接口；`user_id + session_id` 隔离的 SQLite 多轮历史；一个仅返回脱敏 Schema 元数据的 `search_schema` Tool；模型可以直接回复或调用该 Tool，拿到结果后继续推理并回复；可注入的 LLM Adapter；完整自动化测试。用户信息不足时，用普通助手文字追问并结束本轮，下一条消息通过同一 Session 继续，不做阻塞式 ask_user。
>
> 强制约束：禁止实现 NL2SQL、SQL 执行、Workflow、SSE、MCP、Skill、子 Agent、文件/Shell Tool、真实样例数据、向量库和多模型路由。只注册 `search_schema`，设置 Tool 调用与 Agent 轮数上限，所有 Tool 输入输出均用 Pydantic 校验。测试不得调用真实模型 API，必须使用 Fake LLM / 可替换适配器覆盖直接回复、Tool 调用、空结果、Session 连续性、跨用户隔离和 API 主链路。
>
> 请先输出简短实施计划和拟创建的文件清单；确认设计文档无冲突后再按测试先行实施。实现中保持 Route、Service、Agent、Tool、Storage 职责分离，不修改或删除已有注释。完成后运行相关测试，并报告实际验证结果、文件清单和与原项目的对应关系。
