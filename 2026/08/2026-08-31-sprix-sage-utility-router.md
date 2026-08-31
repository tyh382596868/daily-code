---
date: 2026-08-31
topic: infrastructure
source: trending
repo: wang2122/sprix-sage-router
file: sprix_sage.py
permalink: https://github.com/wang2122/sprix-sage-router/blob/bdc9c24f0f7e170d807ffaeb630a478a5810b8d0/sprix_sage.py#L106-L183
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, infrastructure, agent-routing, utility-scoring, trending]
---

# Sprix SAGE router：先列路线，再按效用选队伍 / Sprix SAGE Router: Enumerate Routes, Then Pick by Utility

> **一句话 / In one line**: `route_with_trace` 把 SELF、COLLABORATE、HANDOFF 都变成候选决策，再选约束内效用最高的一条。 / `route_with_trace` turns SELF, COLLABORATE, and HANDOFF into candidate decisions, then chooses the feasible one with highest utility.

## 为什么重要 / Why this matters

多 agent 系统最容易写成一堆 if/else：“能自己做吗？要不要转交？要不要协作？”Sprix SAGE 的可取之处是把这些模式统一成候选路线，每条都算效用、成本、延迟、风险和诊断信息，最后才排序选择。

Multi-agent systems often degrade into nested if/else logic: can I do it myself, should I hand off, should I collaborate? Sprix SAGE is interesting because it normalizes all modes into candidate routes, scores utility, cost, latency, risk, and diagnostics, then sorts.

## 代码 / The code

`wang2122/sprix-sage-router` — [`sprix_sage.py`](https://github.com/wang2122/sprix-sage-router/blob/bdc9c24f0f7e170d807ffaeb630a478a5810b8d0/sprix_sage.py#L106-L183)

```python
def route_with_trace(
    self,
    task: Task,
    bids: Iterable[Bid] | None = None,
    state: ExecutionState | None = None,
) -> RoutingTrace:
    """Route a task and retain feasible alternatives and exclusion reasons."""

    state = state or ExecutionState()
    self._validate_state(task, state)
    self._draw_cache = {}
    self._effective_skill_cache = {}
    self._cost_cache = {}
    self._latency_cache = {}
    bid_map = self._prepare_bids(task, bids)
    hard_exclusions = {
        agent_id: self._hard_exclusion_reasons(agent, task, state)
        for agent_id, agent in self.agents.items()
    }
    hard_eligible = [
        agent_id for agent_id, reasons in hard_exclusions.items() if not reasons
    ]
    if not hard_eligible:
        raise RuntimeError("no authorized and available agent can execute the task")
    eligible = self._prefilter_candidates(task, hard_eligible, bid_map, state)
    prefiltered = tuple(sorted(set(hard_eligible) - set(eligible)))

    candidates: list[RouteDecision] = []
    if self.incumbent_id in eligible:
        self_decision = self._evaluate(Mode.SELF, (self.incumbent_id,), task, bid_map, state)
        candidates.append(self_decision)
        candidates.extend(
            self._beam_collaboration_decisions(task, eligible, bid_map, state)
        )

    for agent_id in eligible:
        if agent_id == self.incumbent_id:
            continue
        decision = self._evaluate(Mode.HANDOFF, (agent_id,), task, bid_map, state)
        candidates.append(decision)

    decisions = [decision for decision in candidates if decision.feasible]
    if not decisions and not self.allow_degraded:
        raise RuntimeError("no feasible route satisfies team-level budget and deadline constraints")
    if decisions:
        best = max(decisions, key=lambda decision: decision.utility)
    else:
        best = min(
            candidates,
            key=lambda decision: (
                decision.diagnostics.get("constraint_violation", math.inf),
                -decision.utility,
            ),
        )
    active = tuple(state.active_agents)
    switched = bool(active) and (best.mode != state.active_mode or set(best.agents) != set(active))
    selected = replace(best, switch_recommended=switched)
    alternatives = tuple(
        sorted(
            (selected if decision is best else decision for decision in candidates),
            key=lambda decision: (
                not decision.feasible,
                decision.diagnostics.get("constraint_violation", 0.0),
                -decision.utility,
            ),
        )
    )
```

## 逐行讲解 / What's happening

1. **第 114-120 行 / Lines 114-120 (fresh routing state)**:
   - 中文: 每次 routing 都重置抽样、技能、成本、延迟缓存，避免旧任务污染新决策。
   - English: Each routing call resets draw, skill, cost, and latency caches so prior tasks do not contaminate the new decision.
2. **第 121-130 行 / Lines 121-130 (hard eligibility)**:
   - 中文: 权限、可用性这类硬约束先过滤；没有合格 agent 就直接失败。
   - English: Hard constraints such as permissions and availability are checked first; no eligible agent means immediate failure.
3. **第 133-145 行 / Lines 133-145 (mode candidates)**:
   - 中文: 自己做、协作、转交都被显式加入 candidate list。
   - English: Self execution, collaboration, and handoff are all added explicitly to the candidate list.
4. **第 147-159 行 / Lines 147-159 (feasible first)**:
   - 中文: 有可行路线就按效用最大选；没有可行路线但允许 degraded，则选违规最轻的一条。
   - English: If feasible routes exist, choose max utility. If none exist but degraded routing is allowed, choose the smallest constraint violation.
5. **第 163-183 行 / Lines 163-183 (trace)**:
   - 中文: 返回的不只是答案，还有备选项、排除原因和预过滤列表，方便调试。
   - English: The result is not only the selected route; it includes alternatives, exclusion reasons, and prefiltered agents for debugging.

## 类比 / The analogy

像调度维修工单：自己修、找同事一起修、转给专家都先报价排队。最后不是凭感觉拍脑袋，而是看权限、时限、费用和成功率。

It is like dispatching a repair ticket: do it yourself, bring a coworker, or send it to a specialist all get quoted first. The final choice follows permissions, deadline, cost, and success probability instead of gut feel.

## 自己跑一遍 / Try it yourself

```python
routes = [
    {"mode": "self", "feasible": True, "utility": 0.42},
    {"mode": "handoff", "feasible": True, "utility": 0.61},
    {"mode": "collab", "feasible": False, "utility": 0.9, "violation": 3},
]
feasible = [r for r in routes if r["feasible"]]
best = max(feasible, key=lambda r: r["utility"])
print(best)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
{'mode': 'handoff', 'feasible': True, 'utility': 0.61}
```

中文: 高效用但不可行的协作路线不会赢过约束内的 handoff。

English: The high-utility but infeasible collaboration route does not beat the feasible handoff.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Kubernetes scheduling** / **Kubernetes scheduling**: 中文: 先过滤不合格节点，再对剩余节点打分。 / English: Filter invalid nodes first, then score the remaining nodes.
- **LLM tool routing** / **LLM tool routing**: 中文: 工具选择也可以先枚举候选，再按权限、成本、成功率排序。 / English: Tool selection can also enumerate candidates, then rank by permissions, cost, and success probability.

## 注意事项 / Caveats / when it breaks

- **效用函数决定行为 / The utility function defines behavior**: 中文: 权重设错，router 会稳定地做错事。 / English: Bad weights make the router consistently choose the wrong thing.
- **trace 不是执行保证 / Trace is not execution**: 中文: 这段代码只负责决策，真正执行还要处理超时、重试和状态回写。 / English: This code decides the route; execution still needs timeout, retry, and state-update handling.

## 延伸阅读 / Further reading

- Sprix SAGE source: https://github.com/wang2122/sprix-sage-router/blob/bdc9c24f0f7e170d807ffaeb630a478a5810b8d0/sprix_sage.py
- Current trending source used for discovery: https://gittrend.io/trending/ai-infrastructure
