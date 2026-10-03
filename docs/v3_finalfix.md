# V3 最终修复：完整上下文预算与动态滚动摘要实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**目标：** 将当前“仅按未摘要消息条数或原始消息 Token 判断压缩”的实现，改为基于完整模型 Prompt 的预算管理：固定上下文最多占软阈值的 20%，当前用户消息不可压缩，最近原始消息最多保留 12 条但按剩余 Token 动态缩小，被移出的旧消息与旧摘要滚动压缩。

**架构：** `ContextManager` 成为唯一的上下文预算决策者：它先加载固定上下文、旧摘要、未摘要原始消息和当前用户消息，计算软阈值与硬预算，再选出可保留的最近消息。`SummaryService` 不再自行决定“是否压缩”，只负责将调用方明确交给它的“旧摘要 + 被移出消息”压缩成新摘要；当最终摘要稳定后，才由 `ContextManager` 调用一次长期记忆提炼。已完成的单文件 `MEMORY.md` 机制保持不变：仅在最终摘要成功后提炼条目，当前轮结束前写入，下一轮整体注入。

**技术边界：** Python、Pydantic、SQLite、LiteLLM。使用现有 `LLMAdapter.complete(messages, tools)`；所有测试使用 FakeLLM 或 mock，不调用真实模型网络。本期不引入 Celery、Redis、向量检索、MCP、Skill、前端改动或数据库删表迁移。

---

## 1. 当前问题与最终目标

当前 `SummaryService.compact_if_needed()` 的压缩判断只计算：

```text
未摘要原始历史消息 + 当前用户消息
```

它没有计入固定 system prompt、`Agent.md`、`MEMORY.md` 和已有会话摘要。并且即使 Token 已超限，只要未摘要消息不超过 12 条，就没有任何消息会被移入摘要。

最终实现必须把“完整 Prompt 是否安全”作为唯一事实来源：

```text
固定 system prompt
+ Agent.md
+ MEMORY.md
+ 会话摘要
+ 动态选择的最近原始消息
+ 当前用户消息
```

其中用户消息绝不静默截断；近期消息的保留数量可小于 12；原始历史数据仍永远保留在 SQLite。

## 2. 预算定义

### 2.1 三个预算值

设：

```text
W = context_window
O = output_reserve
H = W - O
S = H × compact_ratio
F = 固定上下文的实际 Token
U = 当前用户消息的实际 Token
R = 可分配给最近原始消息的 Token
```

则：

```text
硬输入预算 H = W - O
软压缩阈值 S = H × compact_ratio
固定上下文上限 F_cap = S × fixed_context_ratio
最近原始消息预算 R = S - F - U
```

本期正式配置：

```text
compact_ratio = 0.80
fixed_context_ratio = 0.20
recent_message_limit = 12
```

### 2.2 128K 示例

当模型窗口为 128,000、输出预留为 4,096：

```text
H = 128,000 - 4,096 = 123,904
S = 123,904 × 0.80 = 99,123.2
F_cap = 99,123.2 × 0.20 = 19,824.64
```

实现中 Token 是整数；软阈值和固定上限使用向下取整：

```text
soft_input_budget = 99,123
fixed_context_budget = 19,824
```

若固定上下文实际占 10,000 Token，当前用户消息占 1,000 Token：

```text
R = 99,123 - 10,000 - 1,000 = 88,123
```

这里的 20% 是“固定上下文最多可使用的上限”，不是必须浪费的死预留。固定上下文实际较小时，剩余容量自动借给最近消息。

### 2.3 软阈值与硬预算的职责

```text
S：超过后需要主动压缩，目标是最终 Prompt 不高于此值。
H：绝对安全上限，最终发送给模型的 Prompt 不得超过此值。
```

如果模型规格未出现在 `data/model_context_specs.json`，继续使用现有保守默认值：16,384 窗口、2,048 输出预留。该场景同样按上面的公式计算，不能为 128K 模型单独写死逻辑。

## 3. 上下文分层与优先级

### 3.1 固定上下文 F

固定上下文按以下顺序组装：

```text
1. 固定 system prompt
2. <project_rules> Agent.md
3. <long_term_memory> MEMORY.md
4. <conversation_summary> 当前会话摘要
```

它们共同受 `F_cap` 限制。固定 system prompt 不可删除；其余内容的缩减优先级为：

```text
先缩减 MEMORY.md 注入内容
→ 再缩减 Agent.md 注入内容
→ 最后要求 SummaryService 将旧摘要重压缩为更短摘要
```

缩减任何带 XML 样式标签的内容时，必须保留对应的开闭标签，不能使用字符串前半截直接截断导致 `<project_rules>` 或 `<long_term_memory>` 不闭合。

### 3.2 当前用户消息 U

当前用户消息通过 `TokenManager.estimate()` 单独估算，永远完整保留。它不进入本轮摘要，也不因窗口不足而静默截断。

如果“最小固定 system prompt + 当前用户消息”已经超过硬预算 H，抛出一个稳定的 `ContextBudgetExceeded` 业务异常；Chat API 将其转换为 HTTP 413，并提示用户缩短当前问题。不要丢弃用户内容后假装成功。

### 3.3 最近原始消息 R

从摘要边界之后的原始历史中，按时间从新到旧尝试保留：

```text
最多 12 条消息记录
且总 Token 不得超过 R
```

加入下一条更旧消息会使 Token 超过 R 时，停止；未被保留的更旧消息组成 `messages_to_summarize`。因此实际保留数量可以是 12、8、3 或 0，不是固定值。

`role="tool"` 消息不直接进入主模型上下文，但在移动到摘要时要保留其简短 `content`，使摘要能保留“工具得到的重要结论”。

不引入新的“轮次表”或复杂 turn grouping；本期按现有消息记录顺序从最旧侧移动。测试必须覆盖不会将未摘要的旧消息静默丢失这一事实。

## 4. 正确的压缩流程

### 4.1 首次规划

`ContextManager.prepare()` 先加载：旧摘要、摘要边界之后的所有原始消息、固定 system prompt、`Agent.md`、`MEMORY.md` 和当前用户消息。

它先计算固定上下文和当前用户消息 Token，再计算 R。然后从最新原始消息向前选择最多 12 条且不超过 R 的消息。

以下任一条件满足时都需要压缩：

```text
存在未被选入最近窗口的旧消息
或者
完整预演 Prompt 的 Token 超过 S
或者
旧摘要本身超过固定上下文可分配给摘要的额度
```

### 4.2 生成滚动摘要

`SummaryService` 接收由 `ContextManager` 明确给出的 `messages_to_summarize`。摘要提示输入固定为：

```text
旧摘要
+ 本次移出的原始消息（包括工具摘要）
```

摘要成功后：

```text
新摘要写入 session_summaries.summary
summarized_through_message_id 更新为本次移出消息中的最后一个 message_id
```

原始 `messages` 记录不删除。摘要失败时，旧摘要和旧边界不变；本轮只可使用仍能安全放入预算的近期消息，并在最终无法满足硬预算时返回 HTTP 413，而不是无声丢失消息。

### 4.3 摘要后的二次校验

新摘要文本来自模型，长度不能只靠事前猜测。因此摘要成功后必须重新：

```text
读取新摘要
→ 重建完整 Prompt
→ 重新估算 Token
```

处理规则：

1. 最终 Token 小于等于 S：正常发送给 ChatAgent。
2. 大于 S 但小于等于 H：不发送超长历史；继续从最旧侧移出尚未摘要的消息，并进行最多一次额外摘要压缩。
3. 大于 H：先按第 3.1 节缩减 `MEMORY.md`、`Agent.md` 与摘要；仍大于 H 时抛出 `ContextBudgetExceeded`。

每个用户请求最多允许两次摘要模型调用。第二次仍不能产生合规 Prompt 时，明确失败；不得无限摘要循环。

### 4.4 长期记忆时机

保持已实现的单文件长期记忆策略：

```text
最终摘要成功
→ 从最终新摘要提炼 0～若干 MemoryEntry
→ Pydantic 校验
→ 主聊天回答完成后安全 upsert 到 MEMORY.md
→ 下一轮整体注入受限 MEMORY.md
```

若一次请求发生两次摘要，只使用最后一个成功摘要提炼长期记忆，避免针对中间摘要重复写入。Memory 提炼失败不会回滚成功摘要，也不会影响用户主回复。

## 5. 模块职责调整

| 文件 | 改动 | 最终职责 |
|---|---|---|
| `app/memory/models.py` | 修改 | 新增固定上下文比例、最小摘要空间、预算异常与预算规划数据模型；删除已无用途的 `long_term_memory_limit` |
| `app/memory/token_manager.py` | 修改 | 提供硬预算、软阈值、固定上下文上限和单组件估算辅助方法 |
| `app/memory/summary_service.py` | 修改 | 从“自行决定是否压缩”收敛为“压缩调用方传入的消息”；支持摘要最大 Token/字符限制与一次请求内最后摘要的记忆提炼 |
| `app/memory/context_manager.py` | 重构 | 唯一负责完整 Prompt 规划、动态窗口选择、最多两轮压缩、固定上下文裁剪、最终硬预算校验 |
| `app/services/chat_service.py` | 小改 | 映射 `ContextBudgetExceeded` 需要的使用信息；仅对最后成功摘要的候选记忆调用 `upsert()` |
| `app/api/chat.py` | 修改 | 将 `ContextBudgetExceeded` 映射为 HTTP 413；其他既有 503 语义不变 |
| `app/main.py` | 小改 | 注入新的默认 `ContextPolicy`，不改变 Provider 与模型装配方式 |
| `tests/test_token_manager.py` | 扩展 | 验证预算公式和不同模型规格 |
| `tests/test_summary_service.py` | 重写压缩部分 | 验证显式消息压缩、边界推进、失败不覆盖旧摘要 |
| `tests/test_context_manager.py` | 重写 | 验证 20% 固定上限、12 条上限、Token 动态窗口、二次校验和 413 异常 |
| `tests/test_chat_api.py` | 扩展 | 验证 API 413、压缩后的 Memory 写入和跨轮复用 |
| `docs/v1_minimal-agentic-chat-run.md` | 修改 | 写明预算公式、默认数值、三类模型调用及 413 行为 |

不改动 SQLite schema：现有 `messages.id` 与 `session_summaries.summarized_through_message_id` 足够描述边界。也不改动 ToolRegistry、LLMAdapter、模型工厂、前端页面。

## 6. 数据模型和接口约定

### 6.1 ContextPolicy

保留现有：

```text
recent_message_limit = 12
compact_ratio = 0.80
default_context_window = 16384
default_output_reserve = 2048
summary_max_chars = 4000
```

新增：

```text
fixed_context_ratio = 0.20
minimum_summary_tokens = 512
max_summary_calls_per_request = 2
```

限制：

```text
fixed_context_ratio：大于 0、小于 compact_ratio
minimum_summary_tokens：至少 128
max_summary_calls_per_request：只能是 1 或 2
```

删除 `long_term_memory_limit`，因为当前单文件 `MEMORY.md` 整体注入策略不使用 Top-K 条数。

### 6.2 TokenManager

新增只读属性或同等公开方法，语义必须固定：

```text
hard_input_budget = context_window - output_reserve
soft_input_budget = floor(hard_input_budget × compact_ratio)
fixed_context_budget = floor(soft_input_budget × fixed_context_ratio)
```

保留现有 `estimate(messages)` 作为统一 Token 估算入口。不得为了估算 Token 额外发起 `LLMAdapter.complete()`；只使用 LiteLLM `token_counter` 或 UTF-8 字节兜底。

### 6.3 ContextBudgetExceeded

新增应用内异常，至少携带：

```text
estimated_tokens
hard_input_budget
reason
```

它只表达“在不截断当前用户内容、也不静默丢失未摘要会话的前提下无法构造安全 Prompt”。API 对外不暴露内部 Prompt 内容、摘要内容或记忆内容。

### 6.4 SummaryService

将当前 `compact_if_needed(user_id, session_id, current_message)` 拆为职责明确的行为：

```text
compact(user_id, session_id, old_summary, messages_to_summarize, max_summary_chars)
extract_long_term_memory(summary)
```

`compact()` 只在 `messages_to_summarize` 非空，或调用方要求将过长旧摘要重新压缩时调用模型。调用成功后才保存摘要并推进边界；失败绝不覆盖旧摘要。`compact()` 不自动提炼 Memory；`ContextManager` 在所有摘要轮次完成且最终 Prompt 合规后，才调用一次 `extract_long_term_memory()`。该方法保持现有单字段 JSON 约定。

## 7. 实施任务

### Task 1：先建立预算公式与异常的失败测试

**文件：**

- 修改：`app/memory/models.py`
- 修改：`app/memory/token_manager.py`
- 修改：`tests/test_token_manager.py`

- [ ] **Step 1：为 128K 预算公式写失败测试。**

使用 `ContextPolicy(fixed_context_ratio=0.20)` 和 128,000/4,096 的临时模型规格，断言：

```text
hard_input_budget = 123,904
soft_input_budget = 99,123
fixed_context_budget = 19,824
```

再为未知模型断言预算来自默认窗口和默认输出预留；为 65,536/4,096 规格断言软阈值为 49,152、固定上下文预算为 9,830。

- [ ] **Step 2：为配置边界写失败测试。**

断言以下配置被 Pydantic 拒绝：

```text
fixed_context_ratio <= 0
fixed_context_ratio >= compact_ratio
minimum_summary_tokens < 128
max_summary_calls_per_request 不等于 1 或 2
```

同时断言 `long_term_memory_limit` 不再是 `ContextPolicy` 的可接受字段。

- [ ] **Step 3：运行失败测试。**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_token_manager.py -q
```

预期：当前 `ContextPolicy` 与 `TokenManager` 没有新字段和预算属性，测试失败。

- [ ] **Step 4：实现最小预算模型。**

在 `ContextPolicy` 增加第 6.1 节字段，移除未使用的 `long_term_memory_limit`。在 `TokenManager` 增加第 6.2 节预算属性，统一用 `math.floor` 产生整数预算。定义 `ContextBudgetExceeded`，但本任务不接入 API。

- [ ] **Step 5：运行预算测试。**

运行相同命令。预期：预算公式、默认兜底和配置校验全部通过，不产生模型网络请求。

### Task 2：让 SummaryService 只做显式滚动压缩

**文件：**

- 修改：`app/memory/summary_service.py`
- 修改：`tests/test_summary_service.py`

- [ ] **Step 1：写显式消息压缩失败测试。**

创建旧摘要且追加 6 条消息，调用新的 `compact()` 并只传入最旧的 4 条。FakeLLM 返回新摘要。断言：

1. 摘要模型输入含旧摘要和这 4 条消息，不含最近保留的 2 条或当前用户消息。
2. SQLite 只更新一条 summary，边界等于第 4 条被移出消息的 `message_id`。
3. 原始 6 条消息仍全部存在。
4. 摘要调用使用 `tools=[]`。

- [ ] **Step 2：写摘要失败和过长旧摘要重压缩失败测试。**

分别让 FakeLLM 抛 `LLMServiceUnavailable`、返回空内容。断言旧摘要和边界保持不变。再传入空 `messages_to_summarize` 与一个超过传入上限的旧摘要，断言 SummaryService 会请求“仅压缩旧摘要”的调用，成功后边界不倒退。

- [ ] **Step 3：运行失败测试。**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_summary_service.py -q
```

预期：当前方法自行读取数据库并按 12 条做决策，无法满足显式输入契约。

- [ ] **Step 4：重构 SummaryService。**

移除压缩阈值判断和近期窗口判断；这些判断只能位于 ContextManager。实现第 6.4 节 `compact()` 契约。摘要提示继续保留用户目标、事实、约束、未解决事项和工具结论；并按调用方传入的最大字符数约束输出。`compact()` 不自动调用长期记忆提炼；记忆提炼失败时由 ContextManager 处理为空候选，且不影响已成功的最终摘要。

- [ ] **Step 5：运行 SummaryService 测试。**

运行相同命令。预期：显式消息选择、边界推进、失败保护和 Memory 提炼行为通过。

### Task 3：实现完整 Prompt 预算规划和动态窗口

**文件：**

- 修改：`app/memory/context_manager.py`
- 修改：`tests/test_context_manager.py`

- [ ] **Step 1：写 12 条短消息的失败测试。**

构造 128K 模型规格，固定上下文低于 20% 上限，12 条短历史消息和一个短当前问题总量低于软阈值。断言：

1. 不调用 SummaryService。
2. 最终消息含全部 12 条历史和当前问题。
3. 顺序固定为 system、Agent.md、MEMORY.md、summary、近期历史、当前问题。

- [ ] **Step 2：写“12 条超长消息仍必须压缩”的失败测试。**

使用确定性 Token 计数器：固定上下文为 10,000、当前问题为 1,000、12 条历史每条为 10,000 Token。断言：

1. 固定 20% 上限下，ContextManager 不保留全部 12 条。
2. 它从最旧消息开始移出，传给 SummaryService。
3. 最新消息始终优先保留，当前用户消息完整存在。
4. 新摘要边界前进，且最终完整 Prompt 小于等于软阈值。

- [ ] **Step 3：写固定上下文超额的失败测试。**

构造过长 `MEMORY.md`、`Agent.md` 和旧摘要，断言：

1. 固定 system prompt 保留。
2. 首先缩减长期记忆，再缩减项目规则；标签始终成对完整。
3. 若旧摘要仍超出可分配摘要空间，触发一次仅旧摘要的重压缩。
4. 当前用户消息不被截断。

- [ ] **Step 4：写硬预算失败测试。**

构造“最小固定 system prompt + 当前用户消息”仍超过 H 的确定性计数器场景，断言抛出 `ContextBudgetExceeded`，并包含估算值、硬预算和非敏感原因。再构造连续两次摘要后仍超过 H 的场景，断言最多调用两次摘要模型且不会静默省略消息。

- [ ] **Step 5：运行失败测试。**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_context_manager.py -q
```

预期：当前代码在调用 SummaryService 前没有完整 Prompt 预算规划，并固定切最近 12 条，测试失败。

- [ ] **Step 6：实现 ContextManager 预算规划。**

按以下顺序实现：

```text
加载旧摘要、未摘要消息、固定 system、Agent.md、MEMORY.md、当前用户消息
→ 估算 F 与 U
→ 将固定上下文压入 F_cap
→ 计算 R
→ 从最新消息向前选择不超过 12 条且不超过 R 的原始消息
→ 将未选择的最旧消息交给 SummaryService.compact()
→ 重新加载新摘要并重建完整 Prompt
→ 再次估算并在必要时执行最多一次额外压缩
→ 最终 Prompt 合规后，从最后成功摘要提炼一次 MemoryEntry
→ 最终验证 <= H
→ 返回 PreparedContext
```

任何未被保留的原始消息必须同时进入成功摘要，或因 `ContextBudgetExceeded` 明确失败；不得静默消失。`PreparedContext` 继续返回 `maintenance`，以便 ChatService 记录摘要和 Memory usage。

- [ ] **Step 7：运行 ContextManager 测试。**

运行相同命令。预期：动态窗口、固定 20% 上限、滚动摘要、最终硬预算和异常路径全部通过。

### Task 4：接入 API、用量记录与最终文档

**文件：**

- 修改：`app/services/chat_service.py`
- 修改：`app/api/chat.py`
- 修改：`app/main.py`
- 修改：`tests/test_chat_api.py`
- 修改：`docs/v1_minimal-agentic-chat-run.md`

- [ ] **Step 1：写 API 413 失败测试。**

注入确定性 TokenManager，使“固定 system prompt + 当前用户消息”超过硬预算。调用 `POST /api/v1/chat`，断言：

```text
HTTP 413
detail 不包含 Agent.md、MEMORY.md、摘要、历史消息或 API Key
detail 明确要求缩短当前问题
```

并断言没有新增当前用户原始消息、没有覆盖摘要、没有写 Memory。

- [ ] **Step 2：写完整压缩与 Memory 时序测试。**

用 128K/4,096 规格、固定 20% 上限和确定性 Token 计数，连续发送足够多且足够长的消息。FakeLLM 按顺序返回：摘要、新摘要的 Memory JSON、主聊天答复。断言：

1. `session_summaries` 保存新摘要和正确 message ID 边界。
2. `messages` 表原始消息未删除。
3. `token_usages` 包含 `summary`、`memory`、`chat` 三种操作。
4. 响应返回后 `MEMORY.md` 已更新。
5. 下一轮模型输入含受限的 `<long_term_memory>`。
6. 本轮 ChatAgent 收到的新摘要，但不重复收到刚写入的 MemoryEntry。

- [ ] **Step 3：运行 API 失败测试。**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_chat_api.py -q
```

预期：当前 API 没有 413 映射，且当前 ContextManager 不按完整 Prompt 选择动态窗口。

- [ ] **Step 4：最小接入实现。**

`ChatService.chat()` 在调用 `ContextManager.prepare()` 时允许 `ContextBudgetExceeded` 向 API 层传播；只有 prepare 成功后才追加当前 user 消息。保持当前摘要成功后、主聊天答复完成后写 `MEMORY.md` 的时序。`app/api/chat.py` 映射预算异常为 413；不得改变真实模型配置错误的 503 映射。

`main.py` 继续允许 `context_policy` 与 `token_manager` 测试注入；不从环境变量读取新的预算比例。

- [ ] **Step 5：更新运行说明。**

在 V3 章节明确写入：

1. 预算公式 `H = window - output_reserve`、`S = H × 80%`、固定上下文最多 `S × 20%`。
2. 当前用户消息不截断；最近消息最多 12 条，但由剩余 Token 动态决定实际条数。
3. 摘要成功时至少额外产生“摘要调用 + Memory 提炼调用”两次大模型调用；若 Agent 调工具，主聊天可再产生两次模型调用。
4. `MEMORY.md` 是单文件整体注入，下一轮生效。
5. 无法安全构造上下文时 API 返回 413；这不是模型服务故障。

- [ ] **Step 6：运行 API 测试。**

运行相同命令。预期：413、压缩、Memory 时序、usage 和既有 503/工具行为都通过。

### Task 5：全量验证与人工验证

**文件：** 无新增业务文件。

- [ ] **Step 1：运行全量测试和编译检查。**

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m compileall -q app
rg -n "compact_if_needed|long_term_memory_limit|retrieve\(|topics/" app tests
```

预期：pytest 全部通过；`compileall` 无输出；旧的 `compact_if_needed`、`long_term_memory_limit`、主题检索 API 不再出现在业务代码或测试中。

- [ ] **Step 2：人工验证一次 128K 策略。**

用 FakeLLM、临时 SQLite、临时 workspace、128K/4,096 模型规格和低成本确定性 Token 计数器运行以下场景：

1. 12 条短消息：不压缩。
2. 12 条超长消息：保留少于 12 条，旧消息进入新摘要。
3. 旧摘要过长：先重压缩摘要。
4. 当前问题本身过长：返回 413，不写当前用户消息。
5. 摘要成功：下一轮读到新 `MEMORY.md` 条目。

不得使用真实 API Key 或真实模型网络。

## 8. 验收标准

1. 压缩判断基于完整 Prompt，而不是只基于未摘要原始消息。
2. 128K/4,096 配置下，硬预算为 123,904、软阈值为 99,123、固定上下文上限为 19,824 Token。
3. 固定上下文实际较小时，未使用容量自动让给最近消息；不浪费死预留。
4. 最近消息最多 12 条，但 Token 超额时保留数可小于 12；当前用户消息永不静默截断。
5. 被移出近期窗口的原始消息必定进入成功新摘要；SQLite 原始消息不删除，摘要边界用最后被摘要的 `message_id` 推进。
6. 新摘要生成后重新验证完整 Prompt；每请求最多两次摘要调用，不存在无限循环。
7. 最终超过硬预算时 API 返回 413，不伪造成功回复、不泄露内部上下文、不写入当前用户消息。
8. 长期记忆维持单一 `MEMORY.md`：仅摘要成功后提炼，主聊天回答完成后写入，下一轮整体注入。
9. 真实 usage 与估算 Token 继续分别记录；工具调用、503 模型配置错误和用户/Session 隔离语义不变。
10. 全量 pytest 和 `compileall` 通过，测试不调用真实模型网络。

## 9. 交给实施 Agent 的提示词

将以下整段原样发送给实施 Agent：

```text
请在 D:\project\python\myagent 按 docs\v3_finalfix.md 实施“V3 最终修复：完整上下文预算与动态滚动摘要”。当前工作区已经完成单文件 MEMORY.md 长期记忆简化；以现有代码为基线，只实现本文档的完整 Prompt 预算、20% 固定上下文上限、动态最近消息窗口、显式滚动摘要和 API 413。

开始前完整阅读：

- docs\v3_finalfix.md
- docs\v3_context-token-memory-management.md
- docs\v3_memory-md-long-term-memory-fix.md
- app\memory\models.py
- app\memory\token_manager.py
- app\memory\summary_service.py
- app\memory\context_manager.py
- app\memory\long_term_memory.py
- app\memory\project_context.py
- app\storage\session_service.py
- app\services\chat_service.py
- app\api\chat.py
- app\main.py
- tests\test_token_manager.py
- tests\test_summary_service.py
- tests\test_context_manager.py
- tests\test_chat_api.py
- D:\project\qiuzhao\DA\da\agent\node\agentic_node.py
- D:\project\qiuzhao\DA\da\models\session_manager.py
- D:\project\qiuzhao\DA\da\utils\memory_loader.py

必须遵守：

1. 不删除已有注释，不删除或回滚无关改动；当前目录不是 Git 仓库，不执行 commit、reset、checkout 或删除操作。
2. 严格 TDD：每个新行为先补失败测试，再做最小实现；模型调用只用 FakeLLM 或 mock，绝不使用真实 API Key 或真实网络。
3. 压缩决策必须移到 ContextManager。SummaryService 只压缩调用方明确传入的旧摘要和旧消息，不能再自行按 12 条或原始 Token 判断是否压缩。
4. 使用公式：H = context_window - output_reserve；S = floor(H × compact_ratio)；F_cap = floor(S × fixed_context_ratio)。默认 compact_ratio=0.80、fixed_context_ratio=0.20、recent_message_limit=12。
5. 固定上下文包括固定 system prompt、Agent.md、MEMORY.md、会话摘要。当前用户消息不可截断；最近消息最多 12 条，但按 R = S - F - U 的剩余 Token 动态选择，超出的最旧消息必须进入滚动摘要。
6. 每次摘要成功后必须重新计算完整 Prompt；单次请求最多两次摘要调用。若最小固定内容加当前用户消息仍超过硬预算，或两次摘要后仍无法满足硬预算，抛出 ContextBudgetExceeded 并由 API 返回 HTTP 413。不得静默丢失消息或伪造回答。
7. 保持当前单文件 MEMORY.md 机制：只在最后成功摘要后提炼 MemoryEntry；主聊天回答完成后安全写入；下一轮整体注入。不要恢复 topics、Top-K、向量检索或 Celery。
8. 保持 SQLite 原始 messages 永不删除、summary 边界使用 message_id、真实 usage 与估算 Token 分离记录、ToolRegistry 和 LLMAdapter 既有契约不变。
9. 完成后报告：实际修改文件、128K/4,096 时 H/S/F_cap 的数值、动态窗口如何选择消息、摘要边界如何推进、摘要/Memory/主聊天模型调用次数、413 场景、完整测试命令和完整结果。
```
