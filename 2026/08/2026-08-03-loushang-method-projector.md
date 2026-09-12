---
date: 2026-08-03
topic: diffusion
source: trending
repo: zhnt/loushang
file: src/loushang/method/projection.py
permalink: https://github.com/zhnt/loushang/blob/95fa76846ddce541e4c198fe6fa7876bd1835811/src/loushang/method/projection.py#L21-L145
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, trending, agents, methods]
---

# loushang MethodProjector：把方法步骤投影成可执行提示 / loushang MethodProjector: Project a Method Step into Executable Guidance

> **一句话 / In one line**: `MethodProjector` 从结构化 method plan 里抽出当前 step，生成系统指导语和可审计 metadata。 / `MethodProjector` extracts the current step from a structured method plan and emits system guidance plus auditable metadata.

## 为什么重要 / Why this matters

Agent 框架如果只把方法写成自然语言 prompt，就很难追踪“为什么这一轮这样做”。这段代码把指导语、角色、温度和事实元数据分开，让执行和审计走同一份结构化来源。

If an agent framework stores methods only as free-form prompts, it is hard to trace why a turn behaved a certain way. This code separates guidance, role, temperature, and factual metadata while keeping them tied to the same structured source.

## 代码 / The code

`zhnt/loushang` — [`src/loushang/method/projection.py`](https://github.com/zhnt/loushang/blob/95fa76846ddce541e4c198fe6fa7876bd1835811/src/loushang/method/projection.py#L21-L145)

```python
class MethodProjector:
    def project(
        self,
        plan: MethodPlan,
        step: MethodStep,
        context: MethodContext | None = None,
    ) -> MethodProjection:
        _ = context
        content = step.projection.get("content")
        method_content = content if isinstance(content, str) else ""
        return MethodProjection(
            method_id=plan.method_id,
            step_id=step.id,
            system_guidance=f"Use the following method guidance when performing this turn:\n\n{method_content}",
            meta_role=_string_projection_value(step, "meta_role"),
            role_variant=step.role_variant,
            temperature=_float_projection_value(step, "temperature"),
            metadata={
                "plan_facts": _plan_facts(plan),
                "step_facts": _step_facts(plan, step),
                "source_projection": dict(step.projection),
                "source_constraint": dict(step.constraint),
                "source_audit": dict(step.audit),
            },
        )


def _string_projection_value(step: MethodStep, key: str) -> str | None:
    value = step.projection.get(key)
    if isinstance(value, str) and value:
        return value
    return None


def _float_projection_value(step: MethodStep, key: str) -> float | None:
    value = step.projection.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _plan_facts(plan: MethodPlan) -> dict[str, object]:
    return {
        "plan_id": plan.id,
        "method_id": plan.method_id,
        "mode": plan.mode,
        "phase": plan.phase,
        "activity": plan.activity,
        "task": plan.task,
        "metadata": _stable_plan_metadata_facts(plan),
        "applicability": _applicability_facts(plan.applicability),
    }
```

## 逐行讲解 / What's happening

1. **第 29-37 行 / Lines 29-37 (guidance extraction)**:
   - 中文: `content` 只有是字符串才进入 prompt；角色和温度从 projection 字段里安全读取。
   - English: `content` enters the prompt only if it is a string; role and temperature are read safely from projection fields.
2. **第 38-44 行 / Lines 38-44 (trace metadata)**:
   - 中文: 输出不仅有 prompt，还有 plan facts、step facts、原始 projection/constraint/audit。
   - English: The output carries not just a prompt, but plan facts, step facts, and source projection/constraint/audit data.
3. **第 55-61 行 / Lines 55-61 (bool is not float)**:
   - 中文: Python 里 `bool` 是 `int` 子类，所以温度解析要先排除 bool。
   - English: In Python, `bool` is an `int` subclass, so temperature parsing must reject bool first.

## 类比 / The analogy

这像把施工方案投影成当天工单：工人看到的是今天该做什么，主管还能看到工单来自哪份方案、哪个阶段、哪些约束。

It is like projecting a construction plan into today's work order: workers see what to do now, while supervisors can trace the plan, phase, and constraints behind it.

## 自己跑一遍 / Try it yourself

```python
step = {
    "id": "review",
    "projection": {"content": "Check failures first.", "temperature": 0.2},
    "constraint": {"mode": "strict"},
    "audit": {"source": "daily"},
}
plan = {"method_id": "code-review", "phase": "inspect"}
projection = {
    "method_id": plan["method_id"],
    "step_id": step["id"],
    "system_guidance": "Use guidance:\n\n" + step["projection"]["content"],
    "temperature": float(step["projection"]["temperature"]),
    "metadata": {"phase": plan["phase"], "source_audit": step["audit"]},
}
print(projection)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
{'method_id': 'code-review', 'step_id': 'review', 'system_guidance': 'Use guidance:\n\nCheck failures first.', 'temperature': 0.2, 'metadata': {'phase': 'inspect', 'source_audit': {'source': 'daily'}}}
```

中文: 关键不是拼 prompt，而是 prompt 和审计事实一起生成。  
English: The key is not prompt formatting; it is generating guidance and audit facts together.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Workflow engines** / **Workflow engines**: 工作流节点常把 declarative spec 投影成 runtime command。 / Workflow nodes often project declarative specs into runtime commands.
- **Agent policy layers** / **Agent policy layers**: 工具权限和行为约束也常被拆成执行数据和审计数据。 / Tool permissions and behavior constraints are often split into execution data and audit data.

## 注意事项 / Caveats / when it breaks

- **空 content** / **Empty content**: 非字符串 content 会变成空 guidance，需要上游保证方法内容完整。 / Non-string content becomes empty guidance, so upstream must ensure method content is present.
- **metadata 体积** / **Metadata size**: 审计字段太大时会影响日志和 tracing 成本。 / Large audit fields can increase logging and tracing cost.

## 延伸阅读 / Further reading

- loushang `MethodProjector`: https://github.com/zhnt/loushang/blob/95fa76846ddce541e4c198fe6fa7876bd1835811/src/loushang/method/projection.py#L21-L145
