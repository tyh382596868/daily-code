---
date: 2026-09-11
topic: robotics
source: tracked
repo: huggingface/lerobot
file: src/lerobot/annotations/steerable_pipeline/executor.py
permalink: https://github.com/huggingface/lerobot/blob/b6ec0060779550c0a157ae34feb89e0cf86012a8/src/lerobot/annotations/steerable_pipeline/executor.py#L224-L252
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, robotics, annotation-pipeline, staged-data, dependency-order]
---

# LeRobot 计划刷新：事件出现，再回头重跑计划 / LeRobot Plan Refresh: Re-enter the Plan After an Event

> **一句话 / In one line**: LeRobot 把 interjection 当成会改变上下文的事件，在它们出现的时间戳重新发射 plan 行。 / LeRobot treats interjections as context-changing events and re-emits plan rows at their timestamps.

## 为什么重要 / Why this matters

中文：机器人数据标注常常不是一次从头跑到尾的直线流程。中途出现一句插话、一个新事件或一段语音后，原来的计划可能已经过时。这个 executor 没有复制一套新的 prompt 逻辑，而是读取已落盘的 interjection 行，再把时间戳和文本交回原来的 `plan` 模块。

English: Robotics annotation is rarely a simple one-pass pipeline. A spoken interjection or newly detected event can invalidate the plan that was emitted earlier. This executor avoids creating a second prompt implementation: it reads staged interjection rows, extracts their timestamps and text, and sends them back through the original `plan` module.

## 代码 / The code

`huggingface/lerobot` — [`src/lerobot/annotations/steerable_pipeline/executor.py`](https://github.com/huggingface/lerobot/blob/b6ec0060779550c0a157ae34feb89e0cf86012a8/src/lerobot/annotations/steerable_pipeline/executor.py#L224-L252)

```python
    def _run_plan_update_phase(  # noqa: PLR0915
        self, records: list[EpisodeRecord], staging_dir: Path
    ) -> PhaseResult:
        """Re-emit ``plan`` rows at each timestamp the ``interjections`` module produced.

        The ``plan`` module owns the prompt; the ``interjections`` module
        produced the timestamps. This phase therefore calls back into the
        ``plan`` module with the interjection timestamps so its existing
        prompt path is reused.
        """
        if not self.plan.enabled or not self.interjections.enabled:
            return PhaseResult(name="plan_update", episodes_processed=0, episodes_skipped=len(records))
        processed = 0
        for record in records:
            staging = EpisodeStaging(staging_dir, record.episode_index)
            interjection_rows = [
                row for row in staging.read("interjections") if row.get("style") == "interjection"
            ]
            interjection_times = [float(row["timestamp"]) for row in interjection_rows]
            interjection_texts = [str(row.get("content") or "") for row in interjection_rows]
            if interjection_times:
                self.plan.run_plan_updates(record, staging, interjection_times, interjection_texts)
                processed += 1
        # Episodes without any interjections are skipped (no plan refresh
        # needed); count them so the summary's processed+skipped == total.
        return PhaseResult(
            name="plan_update",
            episodes_processed=processed,
            episodes_skipped=len(records) - processed,
        )
```

## 逐行讲解 / What's happening

1. **第 227-231 行 / Lines 227-231**:
   - 中文: 文档字符串先声明所有权边界：`plan` 拥有 prompt，`interjections` 只产生触发时间。
   - English: The docstring establishes ownership: `plan` owns prompt construction, while `interjections` only supplies trigger times.
2. **第 234-235 行 / Lines 234-235**:
   - 中文: 任一模块关闭时，阶段不做副作用，并把所有 episode 记为 skipped。
   - English: If either module is disabled, the phase exits without side effects and counts every episode as skipped.
3. **第 237-243 行 / Lines 237-243**:
   - 中文: 每个 episode 都从 staging tree 读取已经完成的中间结果，再把时间和文本拆成两个并行列表。
   - English: Each episode reads the already-produced staging rows, then separates timestamps and text into parallel lists.
4. **第 244-246 行 / Lines 244-246**:
   - 中文: 只有真的有插话才回调 `run_plan_updates`；这让无事件 episode 不必白跑。
   - English: `run_plan_updates` is called only when an episode has interjections, so event-free episodes avoid unnecessary work.
5. **第 247-252 行 / Lines 247-252**:
   - 中文: `processed + skipped == total` 是一个小但重要的汇总不变量，能让上层报告可信。
   - English: The `processed + skipped == total` invariant keeps the aggregate phase report trustworthy.

## 类比 / The analogy

中文：像厨房的出餐单。主厨先按订单安排步骤；如果服务员突然说某桌过敏，厨房不重写整套菜单系统，而是在那个时间点把原来的出餐计划重新过一遍。

English: Think of a restaurant ticket system. The kitchen starts with an execution plan, then a server reports an allergy. The kitchen does not invent a second menu engine; it re-runs the existing plan logic at the point where the new fact matters.

## 自己跑一遍 / Try it yourself

```python
def refresh_plan(rows):
    events = [r for r in rows if r["kind"] == "interjection"]
    if not events:
        return {"processed": 0, "skipped": 1, "updates": []}
    updates = [(e["time"], e["text"]) for e in events]
    return {"processed": 1, "skipped": 0, "updates": updates}

print(refresh_plan([{"kind": "frame", "time": 0.0}]))
print(refresh_plan([
    {"kind": "frame", "time": 0.0},
    {"kind": "interjection", "time": 2.5, "text": "pick up the red cup"},
]))
```

运行 / Run with:

```bash
python try.py
```

预期输出 / Expected output:

```text
{'processed': 0, 'skipped': 1, 'updates': []}
{'processed': 1, 'skipped': 0, 'updates': [(2.5, 'pick up the red cup')]}
```

中文：值得注意的是，事件阶段只负责产生事实，计划阶段仍然是唯一的解释入口；这让后续维护不容易分叉。

English: The important detail is that the event phase produces facts while the plan phase remains the single interpretation entry point, which prevents logic drift.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **LeRobot PolicyServer** / **LeRobot PolicyServer**: 异步观测进入队列后再按 policy 节拍消费，事件和执行节拍分离。 / Asynchronous observations enter a queue and are consumed at policy cadence, separating event arrival from execution cadence.
- **openpi runtime loop** / **openpi runtime loop**: 每拍重新组合 observation、action 和记录状态，而不是把一次推理当成永恒事实。 / Each tick recomposes observation, action, and logging state instead of treating one inference as permanent truth.
- **streaming ETL DAGs** / **streaming ETL DAGs**: 中间结果落盘后，后续阶段可以基于新增事实重跑局部节点。 / Staged outputs allow downstream nodes to refresh locally when new facts arrive.

## 注意事项 / Caveats / when it breaks

- **staging schema 必须稳定** / **The staging schema must stay stable**: `style`, `timestamp`, and `content` are an implicit contract between phases.
- **回调要幂等** / **The callback should be idempotent**: 重试同一个 episode 时不应重复写出无法区分的计划行。
- **事件顺序要明确** / **Event ordering must be explicit**: timestamps should be sorted or the plan module must define how out-of-order events behave.

## 延伸阅读 / Further reading

- [LeRobot executor.py](https://github.com/huggingface/lerobot/blob/b6ec0060779550c0a157ae34feb89e0cf86012a8/src/lerobot/annotations/steerable_pipeline/executor.py)
- [LeRobot steerable annotation pipeline](https://github.com/huggingface/lerobot/tree/b6ec0060779550c0a157ae34feb89e0cf86012a8/src/lerobot/annotations/steerable_pipeline)
