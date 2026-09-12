---
date: 2026-09-06
topic: infrastructure
source: trending
repo: inclusionAI/AWorld
file: aworld/core/agent/base.py
permalink: https://github.com/inclusionAI/AWorld/blob/main/aworld/core/agent/base.py#L1697-L2244
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, infrastructure, agent-framework, contextvars, lifecycle]
---

# AWorld BaseAgent：把状态机和 task-local 上下文绑在一起 / AWorld BaseAgent: Bind the State Machine to Task-Local Context

> **一句话 / In one line**: 这个 BaseAgent 把状态、reset、contextvars 和 hook 生命周期放进同一个壳里，让并发任务不会互相串台。 / This BaseAgent keeps state, reset, contextvars, and hook lifecycle in one shell so concurrent tasks do not step on each other.

## 为什么重要 / Why this matters

中文：agent 框架最容易出问题的地方不是“能不能跑”，而是“同一个实例被并发复用时，会不会把上下文串掉”。AWorld 把 `AgentStatus.START`、`_agent_context`、`reset()` 和 `async_pre_run()` 串成一个明确的生命周期，状态切换和 task-local 数据都能被追踪。

English: The hard part in an agent framework is not whether it can run, but whether a reused instance leaks context across concurrent tasks. AWorld threads `AgentStatus.START`, `_agent_context`, `reset()`, and `async_pre_run()` into one explicit lifecycle so both state transitions and task-local data stay traceable.

## 代码 / The code

`inclusionAI/AWorld` — [`aworld/core/agent/base.py`](https://github.com/inclusionAI/AWorld/blob/main/aworld/core/agent/base.py#L1697-L2244)

```python
_agent_context: contextvars.ContextVar[Optional['Context']] = contextvars.ContextVar(
    '_agent_context', default=None
)

class BaseAgent(Generic[INPUT, OUTPUT]):
    def __init__(...):
        self.tools = []
        self.state = AgentStatus.START
        self._finished = True

    @staticmethod
    def _get_current_context() -> Optional['Context']:
        return _agent_context.get()

    async def async_run(self, message: Message, **kwargs) -> Message:
        token = _agent_context.set(message.context)
        try:
            await self.async_pre_run(message)
            result = await self.async_policy(...)
            return await self.async_post_run(result, observation, message)
        finally:
            _agent_context.reset(token)

    def reset(self, options: Dict[str, Any] = None):
        self.tools = []
        self.tool_mapping = {}
        self.trajectory = []
        self._finished = True

    async def async_pre_run(self, message: Message):
        if isinstance(message.context, AmniContext):
            message.context.put("start", self.id())
            agent_start_times = message.context.get("agent_start_times") or {}
```

## 逐行讲解 / What's happening

1. **第 1697-1715 行 / Lines 1697-1715**:
   - 中文: agent 初始化时就把工具列表、状态机和 sandbox 边界定好。
   - English: The agent initializes its tool list, state machine, and sandbox boundary up front.
2. **第 1728-1773 行 / Lines 1728-1773**:
   - 中文: `_get_current_context()` 直接从 `contextvars` 里取当前 task 的上下文，避免共享实例互相污染。
   - English: `_get_current_context()` reads the current task’s context straight from `contextvars`, which prevents cross-task contamination.
3. **第 1897-2000 行 / Lines 1897-2000**:
   - 中文: `async_run()` 在进入和退出时都设置/恢复 token，确保上下文只活在这次执行里。
   - English: `async_run()` sets and restores the token on entry and exit, so the context only lives for this execution.
4. **第 2139-2244 行 / Lines 2139-2244**:
   - 中文: `reset()` 和 `async_pre_run()` 把 agent 生命周期重新归零，并给 hook 记录起点时间。
   - English: `reset()` and `async_pre_run()` reinitialize the agent lifecycle and record start time for hooks.

## 类比 / The analogy

中文：像一个共享值班室，门口先发临时门禁卡，离开时再回收；每个人都能进房间，但只能在自己的班次里改白板。

English: It is like a shared duty room where each person gets a temporary badge on entry and hands it back on exit; everyone can enter, but only during their own shift can they edit the board.

## 自己跑一遍 / Try it yourself

```python
import contextvars

current = contextvars.ContextVar("current", default=None)
token = current.set("task-a")
print(current.get())
current.reset(token)
print(current.get())
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
task-a
None
```

中文：AWorld 关心的不是“一个 agent 能不能跑”，而是“多个任务复用同一个 agent 时还能不能各自保持语义清晰”。

English: AWorld cares less about whether one agent can run than about whether one reused agent still keeps each task semantically separate.
