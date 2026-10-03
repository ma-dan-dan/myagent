# V3 长期记忆简化修复计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**目标：** 将已实现的“索引 + topics/ 文件 + 关键词 Top-K 召回”长期记忆，收敛为 DA 当前实现方式的最小版本：会话摘要成功后提取少量长期记忆条目，安全地直接写入单个 `MEMORY.md`；后续聊天把受限的 `MEMORY.md` 整体注入上下文。

**架构：** 短期记忆仍由 SQLite 的原始消息和 `session_summaries` 负责；长期记忆只使用项目级 `.myagent/memory/chat/MEMORY.md`。`SummaryService` 在摘要成功后调用模型提取候选条目，`LongTermMemoryStore` 负责校验、去重、容量限制和原子写入，`ContextManager` 每轮读取受限文件并作为 `<long_term_memory>` 系统消息注入。没有 topics 目录、向量库、关键词排序或 Top-K 检索。

**技术边界：** Python、Pydantic、SQLite、LiteLLM。所有模型调用使用现有 `LLMAdapter.complete(messages, tools)`；测试全部使用 FakeLLM 或 mock，不调用真实网络。

---

## 1. 为什么要这样修复

DA 原项目的 `da/utils/memory_loader.py` 在运行时只读取 `.da/memory/<node>/MEMORY.md`，并按 200 行、25 KB 截断后注入系统提示词。它没有在后端实现“关键词匹配后取 Top 3 主题文件”的检索器。

DA 的 `memory_context_1.0.j2` 还规定：默认直接向 `MEMORY.md` 写简短条目；只有真正很长的专题才拆到独立文件。本项目当前仍处于第一版开放式 Chat 阶段，先只复现前半部分。

当前实现的 `topics/`、frontmatter、`retrieve(query)` 和按问题排序，是超出本阶段理解目标的复杂度。本修复将其移除，保留“摘要成功后沉淀、下一轮复用、严格容量限制和安全写入”这条最小闭环。

## 2. 本期范围

### 2.1 要实现

1. 只使用 `.myagent/memory/chat/MEMORY.md` 保存长期记忆。
2. 每次增量摘要成功后，使用一次无工具模型调用，从该摘要中提取候选长期记忆。
3. 候选记忆采用严格 JSON + Pydantic 校验；无候选时返回空数组，不写文件。
4. 记忆文件按“一个条目一行”的 Markdown 列表保存，去除完全重复条目。
5. 每轮聊天加载整份受限 `MEMORY.md`，包装为 `<long_term_memory>` 系统消息注入。
6. 写入不超过 200 行、25 KB；新增条目放不下时跳过新增内容，绝不截断或破坏已有条目。
7. 读取失败、提炼失败、校验失败或写入失败，都不能阻断当前用户聊天，也不能覆盖已有文件。

### 2.2 明确不做

1. 不创建、读取或删除 `topics/` 目录。
2. 不实现关键词检索、向量检索、Rerank 或 Top-K。
3. 不让模型直接修改 `Agent.md`。
4. 不把原始对话、工具原始结果、密码、API Key、认证 Token 或个人敏感信息写入长期记忆。
5. 不实现“某位用户独享的长期记忆”。本期 `MEMORY.md` 是工作区级项目知识，不能存用户私密信息；如未来需要用户偏好隔离，另立版本按 `user_id` 分目录实现。
6. 不处理旧 `topics/` 内容的自动迁移或删除。当前工作区的 `.myagent/memory/chat/MEMORY.md` 为空；实现不得执行删除操作。

## 3. 目标数据格式与调用流

### 3.1 MEMORY.md 格式

文件只保存标题和一行一个的条目。首次写入时可保留一个固定标题；后续只追加或保留已存在的条目。

```markdown
# 长期记忆

- 用户希望默认使用中文解释代码概念。
- 当前项目通过 LLM_PROVIDER 选择模型厂商，模型名在 Provider 类常量中配置。
```

每个条目必须是单行、非空、最大 300 个字符。条目内容中不允许换行、Markdown 文件链接、目录路径或工具原始输出。

### 3.2 记忆提炼模型契约

用户给出的自然语言提炼规则保留，但实现不能依赖模型输出 Markdown 列表。为了继续使用 `json.loads()` 与 Pydantic 校验，提示词必须要求模型仅输出 JSON 数组。

语义要求：

1. 仅从“本次新生成的会话摘要”提取稳定、可跨会话复用的项目知识或明确的长期协作偏好。
2. 不提取一次性任务、普通闲聊、原始对话、未经确认推测、工具数据、私人信息或任何凭据。
3. 每个候选是一句事实准确的短句，最多 300 字。
4. 没有候选时输出空数组。

JSON 元素只包含 `content` 字段，例如：

```json
[
  {"content": "用户希望默认使用中文解释代码概念。"}
]
```

`MemoryEntry` 相应简化为只含 `content` 的 Pydantic 模型。不要保留 `topic_id`、`title`、`keywords` 字段，因为本期没有主题文件和检索索引。

### 3.3 完整时序

```text
历史消息超过短期记忆阈值
        ↓
SummaryService 成功生成并保存增量摘要
        ↓
SummaryService 以该摘要请求候选长期记忆（tools=[]）
        ↓
JSON 解析 + Pydantic 校验 + 安全校验
        ↓
ChatService 在本轮聊天结果完成后调用 LongTermMemoryStore.upsert()
        ↓
安全地更新 MEMORY.md，并记录 memory usage
        ↓
下一轮请求：ContextManager.load() 整份受限 MEMORY.md
        ↓
作为 <long_term_memory> 注入模型上下文
```

本轮正在进行的 Chat 不要求立刻看到新写入的长期记忆；它从下一轮开始复用。这避免在当前请求中重复注入刚从同一段摘要提炼出的信息。

## 4. 文件变更与职责

| 文件 | 变更 | 职责 |
|---|---|---|
| `app/memory/models.py` | 修改 | 将 `MemoryEntry` 收敛为单字段 `content`，保留 `CompactionResult.memory_entries` 列表结构 |
| `app/memory/long_term_memory.py` | 重写 | 直接读写单个 `MEMORY.md`；安全校验、去重、200 行/25 KB 限制、原子写入和系统消息包装 |
| `app/memory/summary_service.py` | 修改 | 将长期记忆提炼提示与 JSON 校验改为单字段条目 |
| `app/memory/context_manager.py` | 修改 | 不再按当前问题调用 `retrieve(query)`；改为加载整个受限 `MEMORY.md` |
| `app/services/chat_service.py` | 小改 | 保留摘要成功后 `upsert()` 的时机；删除不再需要的主题检索假设 |
| `tests/test_long_term_memory.py` | 重写 | 验证直接文件写入、去重、安全、容量和读取注入 |
| `tests/test_summary_service.py` | 修改 | 验证 JSON 候选条目解析与失败降级 |
| `tests/test_context_manager.py` | 修改 | 验证整份长期记忆注入，而非按 query 召回 |
| `tests/test_chat_api.py` | 修改 | 验证一次真实压缩后的记忆写入及下一轮复用 |
| `docs/v1_minimal-agentic-chat-run.md` | 修改 | 更新长期记忆运行说明，删除 topics/Top-K 描述 |

不修改 SQLite 表、`Agent.md`、`ChatAgent`、ToolRegistry、LLMAdapter 工厂或前端。

## 5. 关键实现规则

### 5.1 LongTermMemoryStore 的公开行为

保留类名 `LongTermMemoryStore`，减少无关重命名。其公开行为收敛为：

1. `load()`：读取 `MEMORY.md`，按 200 行和 25 KB 截断；文件不存在、空文件或读取错误时返回空字符串；只读，不创建或修改文件。
2. `as_system_message()`：`load()` 有内容时返回一个包含 `<long_term_memory>` 的 system message；无内容时返回 `None`。
3. `upsert(entries)`：校验每个条目，跳过不安全、重复或超出容量的条目；将真正写入的条目列表返回给调用方。

首次有有效条目写入时，才创建 `.myagent/memory/chat/` 和 `MEMORY.md`。不要在应用启动或每次 `create_app()` 时因为读取长期记忆而创建空目录。

### 5.2 安全与容量

1. 对每个 `MemoryEntry.content` 先 `strip()`，拒绝空白、多行内容、超过 300 字符的内容。
2. 拒绝常见凭据模式：`api_key`、`password`、`secret`、`cookie`、`authorization`、`Bearer `、`sk-`，以及带值的 `access_token`、`refresh_token`、`api_token`。不要用裸词 `token` 作为拒绝条件，否则“Token 用量管理”这类正常项目知识无法保存。
3. 只接受 Markdown 普通文本条目；拒绝路径穿越、链接写法和控制字符。
4. 去重使用规范化后的完整条目文本：去首尾空白、压缩连续空格、英文转小写后比较。相同条目只能写一次。
5. 文件容量判断必须发生在写入前。若下一条会使文件超过任一硬上限，跳过该条，继续处理其余候选，不截断旧条目。
6. 使用同目录临时文件加原子 `replace()` 更新 `MEMORY.md`。写失败时保留旧文件原状，并将失败吞掉为日志，不传播到聊天接口。

### 5.3 ContextManager 注入规则

最终模型上下文顺序保持为：固定系统提示词 → `Agent.md` 项目规则 → `MEMORY.md` 长期记忆 → 会话摘要 → 最近原始消息 → 当前用户消息。

长期记忆读取不依赖当前用户问题，不执行关键词检索，也不读取 topics 文件。受限 `MEMORY.md` 为空时不添加 `<long_term_memory>` 消息。

### 5.4 与 Token 预算的关系

`MEMORY.md` 的 200 行/25 KB 是文件级上限，不等于本轮模型 Token 预算。`ContextManager` 仍必须把长期记忆算入最终 Token 估算，并由已有上下文预算逻辑处理过长输入。

本计划不修复已审查发现的“最终上下文可能仍超过 Token 预算”问题；该问题应单独形成一份上下文预算修复计划，避免与本期长期记忆简化混在一起。

## 6. 实施任务

### Task 1：先用测试锁定单文件长期记忆行为

**文件：**

- 修改：`app/memory/models.py`
- 重写：`app/memory/long_term_memory.py`
- 重写：`tests/test_long_term_memory.py`

- [ ] **Step 1：写直接写入 MEMORY.md 的失败测试。**

在临时 workspace 创建 `LongTermMemoryStore`，写入两个合法候选条目。断言：

1. 只创建 `.myagent/memory/chat/MEMORY.md`，不创建 `topics/`。
2. 文件存在固定标题和两条 `- ` 开头的记忆。
3. `load()` 返回这两条记忆。
4. `as_system_message()` 返回 system role，并包含 `<long_term_memory>` 包装。

- [ ] **Step 2：写去重、安全和容量失败测试。**

覆盖以下场景：

1. 同一条内容（仅英文大小写或多余空格不同）第二次 `upsert()` 不再写入。
2. `api_key=...`、`Bearer ...`、`access_token=...`、多行字符串和超过 300 字的条目全部被拒绝。
3. 先写入合法旧条目，再尝试不安全条目，断言旧内容保持不变。
4. 用小型可注入容量限制构造 store，验证会超出 200 行或 25 KB 限制的新增条目被跳过，已有文件未截断。
5. 手工创建超过 200 行或 25 KB 的 `MEMORY.md`，断言 `load()` 仅返回受限内容且不会修改源文件。

- [ ] **Step 3：运行失败测试。**

运行：

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_long_term_memory.py -q
```

预期：旧的 `topic_id/title/keywords` 数据模型、`topics/` 写入和 `retrieve(query)` 实现与新测试不兼容。

- [ ] **Step 4：实现最小单文件存储。**

将 `MemoryEntry` 改为单字段模型，并实现第 5.1 与 5.2 节中的 `load()`、`as_system_message()` 和 `upsert(entries)` 行为。复用项目当前的原子写入方式，但目录创建只能发生在首次成功写入时。不得删除已有注释。

- [ ] **Step 5：运行单文件记忆测试。**

运行相同命令。预期：所有长期记忆测试通过，且测试目录内没有 `topics/`。

### Task 2：调整摘要提炼契约和上下文注入

**文件：**

- 修改：`app/memory/summary_service.py`
- 修改：`app/memory/context_manager.py`
- 修改：`tests/test_summary_service.py`
- 修改：`tests/test_context_manager.py`

- [ ] **Step 1：写摘要提炼 JSON 契约失败测试。**

让 FakeLLM 返回一个只含 `content` 的 JSON 数组，断言 `extract_long_term_memory()` 解析为候选条目。再分别返回空数组、非 JSON、含额外字段，断言它们不会产生候选且不会抛到聊天路径。凭据模式的拒绝由 `LongTermMemoryStore.upsert()` 负责，并已在 Task 1 覆盖；不能把安全逻辑复制到两层。

- [ ] **Step 2：写完整 MEMORY.md 注入失败测试。**

在临时 workspace 写入两条长期记忆，准备一次 ContextManager 调用。断言：

1. 输出顺序仍是固定系统提示词、项目规则、长期记忆、会话摘要、近期消息、当前消息。
2. `<long_term_memory>` 中同时包含两条记忆。
3. 当前用户问题改变时，长期记忆内容仍相同；证明不存在 query 关键词筛选。
4. 空文件时不生成长期记忆 system message。

- [ ] **Step 3：运行失败测试。**

运行：

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_summary_service.py tests\test_context_manager.py -q
```

预期：旧的 `MemoryEntry` 字段、`retrieve(current_message)` 和主题检索断言导致失败。

- [ ] **Step 4：修改 SummaryService。**

将长期记忆提炼提示改为第 3.2 节的 JSON 契约。保持摘要调用和记忆提炼调用都使用 `tools=[]`；保持解析、Pydantic 校验或模型调用失败时返回空候选和 usage，而不是中断用户聊天。

- [ ] **Step 5：修改 ContextManager。**

删除 `retrieve(current_message)` 调用，改用 `LongTermMemoryStore.as_system_message()`。不要改动短期摘要边界、SQLite 消息读取顺序、当前用户消息最后追加的行为。

- [ ] **Step 6：运行摘要与上下文测试。**

运行相同命令。预期：全部通过，且不访问真实模型网络。

### Task 3：验证“压缩后写入、下一轮复用”的完整链路

**文件：**

- 修改：`app/services/chat_service.py`（仅清理不再存在的 topics/检索假设，如无此类代码则不改）
- 修改：`tests/test_chat_api.py`
- 修改：`docs/v1_minimal-agentic-chat-run.md`

- [ ] **Step 1：写 API 集成失败测试。**

使用临时 SQLite、临时 workspace、`recent_message_limit=2` 和有固定返回顺序的 FakeLLM，连续发送三轮消息。第三轮的 `prepare()` 应触发摘要：FakeLLM 依次返回摘要文本、候选记忆 JSON、第三轮普通聊天答复。

断言：

1. 第三轮结束后 `MEMORY.md` 含候选记忆。
2. 第四轮模型输入含 `<long_term_memory>` 与该记忆文本。
3. `MEMORY.md` 只在摘要成功后写入；让摘要失败或候选 JSON 不合法时，文件不存在或保持旧内容，聊天接口仍返回成功响应。
4. 响应仍含既有 `usage.current_turn` 和 `usage.session_total`，并且 `memory` 调用 usage 被计入 Session 总量。

- [ ] **Step 2：运行 API 失败测试。**

运行：

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_chat_api.py -q
```

预期：旧行为仍按当前问题调用 Top-K 检索，无法满足“整个 MEMORY.md 注入”和“无 topics”的断言。

- [ ] **Step 3：完成 ChatService 的最小适配。**

保留现有时序：先 `ContextManager.prepare()`，再保存当前 user 消息、运行 ChatAgent、保存结果、记录 usage；只有 `prepared.maintenance.compacted` 为真时，才调用 `LongTermMemoryStore.upsert()`。不得将长期记忆写入时机提前到摘要成功前，也不得让长期记忆写入失败影响 ChatResponse。

- [ ] **Step 4：更新运行文档。**

将长期记忆说明改为：

1. 唯一文件是 `.myagent/memory/chat/MEMORY.md`，不再有 `topics/` 和关键词 Top-K。
2. 文件只在摘要成功后由安全候选条目更新；`Agent.md` 永远只读。
3. 下一轮请求会把受限的整份 `MEMORY.md` 注入模型；文件级限制是 200 行/25 KB。
4. 这是工作区项目知识，不能保存用户私密信息或凭据。

- [ ] **Step 5：运行 API 测试。**

运行相同命令。预期：API 测试通过，且所有模型响应均由 FakeLLM 提供。

### Task 4：全量回归与人工检查

**文件：** 无新增业务文件。

- [ ] **Step 1：运行全量验证。**

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m compileall -q app
rg -n "topics/|retrieve\(|topic_id|keywords|long_term_memory" app tests docs
```

预期：pytest 全部通过；`compileall` 无输出；前四个旧主题检索关键字不再出现在长期记忆业务代码或测试中。`long_term_memory` 仍应出现在 ContextManager、文档和测试中。

- [ ] **Step 2：人工检查工作区文件。**

在不输出 API Key、不调用真实模型的前提下，用集成测试生成的临时 workspace 确认：长期记忆只有单个 `MEMORY.md`；该文件包含一行一个的条目；不存在测试创建的 `topics/`；摘要失败时原文件未改变。

## 7. 验收标准

1. 项目长期记忆只使用一个 `MEMORY.md`，不维护 topics、frontmatter、索引或 Top-K。
2. 仅在会话摘要成功后提炼长期记忆；提炼失败不会影响摘要、聊天回复或已有文件。
3. 长期记忆条目通过 JSON、Pydantic、安全校验、去重和容量校验后才写入。
4. `MEMORY.md` 中每条记忆一行，文件不超过 200 行和 25 KB；超限新增被安全跳过。
5. 下一轮模型上下文中包含受限的整份 `MEMORY.md`，而不是根据当前问题筛选的若干主题。
6. 原始 SQLite 会话、摘要边界、Token usage、ToolRegistry、LLMAdapter 和 `Agent.md` 只读规则保持不变。
7. 所有测试和 Python 编译检查通过，且测试不请求真实模型网络。

## 8. 交给实施 Agent 的提示词

将下面整段原样发送给实施 Agent：

```text
请在 D:\project\python\myagent 按 docs\v3_memory-md-long-term-memory-fix.md 实施 V3 长期记忆简化修复。本次只把当前 topics/关键词 Top-K 长期记忆改为单个 MEMORY.md 的直接条目模式；不要顺带修改上下文预算、SQLite 短期记忆、ToolRegistry、LLMAdapter、模型配置、前端或 API 路由。

开始前完整阅读：

- docs\v3_memory-md-long-term-memory-fix.md
- docs\v3_context-token-memory-management.md
- app\memory\models.py
- app\memory\long_term_memory.py
- app\memory\summary_service.py
- app\memory\context_manager.py
- app\services\chat_service.py
- tests\test_long_term_memory.py
- tests\test_summary_service.py
- tests\test_context_manager.py
- tests\test_chat_api.py
- D:\project\qiuzhao\DA\da\utils\memory_loader.py
- D:\project\qiuzhao\DA\da\prompts\prompt_templates\memory_context_1.0.j2

必须遵守：

1. 不删除已有注释，不删除或回滚无关改动；当前目录不是 Git 仓库，不执行 commit、reset、checkout 或删除操作。
2. 按 TDD 执行：先补失败测试，再做最小实现；所有模型调用使用 FakeLLM 或 mock，不使用 API Key 或真实模型网络。
3. 长期记忆只使用 .myagent\memory\chat\MEMORY.md。不要创建或读取 topics/，不要实现关键词、向量、Top-K 或重排。
4. 摘要成功后才从摘要提炼 JSON 候选记忆；候选只含 content，一个条目一行。没有候选时输出或处理为空数组。
5. 长期记忆文件是项目级共享知识，不得存用户私密信息、原始对话、工具原始数据、密码、Key、Cookie、Bearer 凭据或认证 Token。
6. MEMORY.md 读取和写入遵守 200 行、25 KB 上限；写入前去重和安全校验，写失败保持旧文件；读取失败不能阻断聊天。
7. 下一轮请求应整体注入受限的 MEMORY.md，不根据当前问题筛选条目。Agent.md 继续只读。
8. 不能修改 V3 上下文预算策略；该问题不属于本次修复。

完成后报告：实际修改文件、候选记忆从摘要到 MEMORY.md 再到下一轮 Prompt 的完整数据流、删除了哪些旧 topics/Top-K 行为、全部测试命令与完整结果、以及未做的范围。
```
