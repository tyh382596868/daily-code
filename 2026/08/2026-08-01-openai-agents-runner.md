---
date: 2026-08-01
topic: infrastructure
source: trending
repo: openai/openai-agents-python
file: src/agents/run.py
permalink: https://github.com/openai/openai-agents-python/blob/main/src/agents/run.py#L253-L329
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, infrastructure, agents, runner]
---

# OpenAI Agents Runner：同步 API 只是异步循环外壳 / OpenAI Agents Runner: The Sync API Is a Shell Around the Async Loop

> **一句话 / In one line**: `Runner.run_sync` 新建事件循环并调用异步 `run`，所以核心 agent 编排只维护一套 async 实现。 / `Runner.run_sync` creates an event loop and calls async `run`, so the agent orchestration keeps one async implementation.

## 为什么重要 / Why this matters

工具调用、模型请求、流式事件和 handoff 都天然是异步任务。库仍然需要给脚本用户一个同步入口；好的做法是让同步 API 变成薄 wrapper，而不是复制一份控制流。

Tool calls, model requests, streaming events, and handoffs are naturally asynchronous. A library still needs a synchronous entry point for script users; the clean design is a thin wrapper, not a duplicated control flow.

## 代码 / The code

`openai/openai-agents-python` — [`src/agents/run.py`](https://github.com/openai/openai-agents-python/blob/main/src/agents/run.py#L253-L329)

```python
class Runner:
    @classmethod
    async def run(
        cls,
        starting_agent: Agent[TContext],
        input: str | list[TResponseInputItem],
        *,
        context: TContext | None = None,
        max_turns: int = DEFAULT_MAX_TURNS,
        hooks: RunHooks[TContext] | None = None,
        run_config: RunConfig | None = None,
        previous_response_id: str | None = None,
    ) -> RunResult:
        runner = DEFAULT_AGENT_RUNNER
        return await runner.run(
            starting_agent,
            input,
            context=context,
            max_turns=max_turns,
            hooks=hooks,
            run_config=run_config,
            previous_response_id=previous_response_id,
        )

    @classmethod
    def run_sync(
        cls,
        starting_agent: Agent[TContext],
        input: str | list[TResponseInputItem],
        *,
        context: TContext | None = None,
        max_turns: int = DEFAULT_MAX_TURNS,
        hooks: RunHooks[TContext] | None = None,
        run_config: RunConfig | None = None,
        previous_response_id: str | None = None,
    ) -> RunResult:
        """Run a workflow synchronously."""
        return asyncio.get_event_loop().run_until_complete(
            cls.run(
                starting_agent,
                input,
                context=context,
                max_turns=max_turns,
                hooks=hooks,
                run_config=run_config,
                previous_response_id=previous_response_id,
            )
        )
```

## 逐行讲解 / What's happening

1. **第 253-276 行 / Lines 253-276 (`run`)**:
   - 中文: classmethod 不自己编排 agent，而是把参数转交给默认 runner 实例。
   - English: The classmethod does not orchestrate agents itself; it forwards arguments to the default runner instance.
2. **第 279-294 行 / Lines 279-294 (`run_sync` signature)**:
   - 中文: 同步入口保留和异步入口几乎一样的参数，调用者不用学两套 API。
   - English: The sync entry point keeps nearly the same arguments as the async entry point, so callers do not learn two APIs.
3. **第 296-308 行 / Lines 296-308 (`run_until_complete`)**:
   - 中文: 真正的控制流仍然是 `cls.run(...)`；同步 API 只负责把 coroutine 跑完。
   - English: The real control flow is still `cls.run(...)`; the sync API only drives the coroutine to completion.

## 类比 / The analogy

这像自动挡汽车的换挡杆：驾驶员看到的是一个简单档位，但底下仍然是同一套发动机和变速箱在工作。

It is like an automatic car shifter: the driver sees a simple lever, but the same engine and transmission do the real work underneath.

## 自己跑一遍 / Try it yourself

```python
import asyncio

async def run(name):
    await asyncio.sleep(0)
    return f"done:{name}"

def run_sync(name):
    return asyncio.get_event_loop().run_until_complete(run(name))

print(run_sync("agent"))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
done:agent
```

同步函数没有复制 async 逻辑，它只是把同一个 coroutine 放进事件循环。

The sync function does not copy async logic; it puts the same coroutine into an event loop.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **HTTP clients** / **HTTP clients**: 很多库维护 async core，再给脚本用户暴露同步包装。 / Many libraries keep an async core and expose a synchronous wrapper for scripts.
- **training launchers** / **training launchers**: CLI 入口通常只解析参数，然后调用同一个内部 runner。 / CLI entry points often parse arguments and call the same internal runner.

## 注意事项 / Caveats / when it breaks

- **已有事件循环时要小心** / **Be careful inside an existing event loop**: notebook 或 async Web server 里再 `run_until_complete` 可能报错。 / Calling `run_until_complete` inside notebooks or async web servers may fail.
- **错误传播不变** / **Errors still propagate**: wrapper 不吞异常，async core 抛出的错误会回到同步调用者。 / The wrapper does not swallow errors; exceptions from the async core return to the sync caller.

## 延伸阅读 / Further reading

- [OpenAI Agents Python runner source](https://github.com/openai/openai-agents-python/blob/main/src/agents/run.py#L253-L329)

