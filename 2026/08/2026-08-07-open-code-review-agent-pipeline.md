---
date: 2026-08-07
topic: infrastructure
source: trending
repo: alibaba/open-code-review
file: internal/agent/agent.go
permalink: https://github.com/alibaba/open-code-review/blob/main/internal/agent/agent.go#L216-L320
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, infrastructure, code-review, agent]
---

# Open Code Review pipeline：确定性工程先切范围，agent 再深读 / Open Code Review Pipeline: Deterministic Scope First, Agent Deep-Reads Later

> **一句话 / In one line**: `Run` 把一次 review 拆成 diff 解析、文件过滤、并发子任务、manifest 收尾四个硬边界。 / `Run` splits a review into diff parsing, file filtering, concurrent subtasks, and manifest finalization.

## 为什么重要 / Why this matters

代码审查 agent 最大的问题不是不会读代码，而是不稳定：漏文件、行号漂移、上下文太大。Open Code Review 的设计把确定性流程放在前面，先用工程代码固定输入范围和 session，再把需要判断的部分交给 LLM loop。

The main problem with code-review agents is not that they cannot read code; it is instability: missed files, drifting line positions, oversized context. Open Code Review puts deterministic plumbing first, fixing inputs and sessions before handing judgment-heavy work to the LLM loop.

## 代码 / The code

`alibaba/open-code-review` — [`internal/agent/agent.go`](https://github.com/alibaba/open-code-review/blob/main/internal/agent/agent.go#L216-L320)

```go
func (a *Agent) Run(ctx context.Context) ([]model.LlmComment, error) {
    if err := a.loadDiffs(ctx); err != nil {
        return nil, fmt.Errorf("load diffs: %w", err)
    }
    a.injectDiffMap()
    a.args.Tools.Freeze()
    a.diffs = a.filterDiffs(a.diffs)
    if len(a.diffs) == 0 {
        return []model.LlmComment{}, nil
    }
    comments, err := a.dispatchSubtasks(ctx)
    err = errors.Join(err, a.finalizeManifest())
    return comments, err
}
```

## 逐行讲解 / What's happening

1. **第 216-245 行 / Lines 216-245 (diff parse)**:
   - 中文: 第一阶段只解析 diff，并把文件数、插入/删除行数写入 telemetry。
   - English: The first phase only parses diffs and records file/line metrics.
2. **第 246-255 行 / Lines 246-255 (frozen context)**:
   - 中文: 构建只读 diff map，冻结工具注册表，再过滤真正要 review 的文件。
   - English: It builds a read-only diff map, freezes the tool registry, then filters reviewable files.
3. **第 256-270 行 / Lines 256-270 (empty review)**:
   - 中文: 没有支持文件时仍然 finalize session，避免运行记录丢失。
   - English: With no supported files, it still finalizes the session so the run is recorded.
4. **第 298-320 行 / Lines 298-320 (dispatch and finalize)**:
   - 中文: LLM 子任务并发执行，最后把 manifest/session 收尾错误和 review 错误合并。
   - English: LLM subtasks run concurrently, and manifest/session finalization errors are joined with review errors.

## 类比 / The analogy

像手术室流程：护士先核对病人、部位和器械，医生再做判断。判断可以复杂，但入口必须机械可靠。

It is like an operating-room checklist: staff verify patient, site, and instruments before the surgeon makes complex judgments. The reasoning can be flexible; the entry process must be mechanical.

## 自己跑一遍 / Try it yourself

```python
files = ["a.py", "README.md", "b.go"]
supported = [f for f in files if f.endswith((".py", ".go"))]
tools_frozen = True
print(supported, tools_frozen)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
['a.py', 'b.go'] True
```

先固定范围，再让智能部分工作。

Fix the scope first, then let the intelligent part work.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **CI review bots** / **CI review bots**: diff 获取、文件过滤和发布评论通常是确定性代码。 / Diff loading, file filtering, and comment publishing are usually deterministic code.
- **Agent tool registries** / **Agent tool registries**: 运行开始后冻结工具集合，避免中途改变能力边界。 / Tool sets are frozen after startup so capability boundaries do not change mid-run.

## 注意事项 / Caveats / when it breaks

- **过滤规则过窄会漏审** / **Over-narrow filters miss files**: 确定性流程稳定，但规则本身要维护。 / Deterministic plumbing is stable, but its rules still need maintenance.
- **manifest 也是产品行为** / **The manifest is product behavior**: 成功、跳过、预算中断都要能复盘。 / Success, skip, and budget stop states must be auditable.

## 延伸阅读 / Further reading

- [Open Code Review agent pipeline](https://github.com/alibaba/open-code-review/blob/main/internal/agent/agent.go#L216-L320)
- [Open Code Review README](https://github.com/alibaba/open-code-review)

