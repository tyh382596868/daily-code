---
date: 2026-09-30
topic: infrastructure
source: trending
repo: muellerberndt/cadence
file: src/cadence/bootstrap.py
permalink: https://github.com/muellerberndt/cadence/blob/aa9feeabae1d30dc6d1f1b621901162590948a74/src/cadence/bootstrap.py#L52-L192
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, infrastructure, training-loop]
---

# Cadence bootstrap：训练循环先做 readiness check，再提交 witness / Cadence Bootstrap: Check Readiness Before Committing Witnesses

> **一句话 / In one line**: 这个 bootstrap helper 把训练写成“评估-观察-再评估”的闭环，并且每次只提交合格的经验。 / This bootstrap helper writes training as an evaluate-observe-evaluate loop, committing only qualified experience.

## 为什么重要 / Why this matters

很多 agent/learning 框架把训练循环藏在大对象里。Cadence 的 `bootstrap` 很小，却把几个工程习惯讲清楚：输入样本先验证，ready check 不写入状态，observe 失败立即停，报告里保留 work、history 和 failure。

Many agent and learning frameworks hide the training loop inside a large object. Cadence’s `bootstrap` is small but explicit: validate samples first, keep readiness checks state-free, stop on observe refusal, and return work, history, and failure details.

## 代码 / The code

`muellerberndt/cadence` — [`src/cadence/bootstrap.py`](https://github.com/muellerberndt/cadence/blob/aa9feeabae1d30dc6d1f1b621901162590948a74/src/cadence/bootstrap.py#L52-L192)

```python
def bootstrap(
    brain, examples, *, checks, max_error, epochs=20, seed=0, budget=None, batch_size=1
):
    """Replay supervised witnesses until target-free readiness checks pass.

    ``examples`` and ``checks`` are nonempty finite sequences of
    ``(input_mapping, target_mapping)`` pairs. All samples are validated and
    copied before work. Only examples are admitted; checks are repeatedly used
    for readiness, so they are a development set, not an untouched final test.

    A local seeded RNG shuffles examples each epoch. Before teaching and after
    each complete epoch, pure ``settle`` queries measure maximum absolute error
    against each unique targeted patch coordinate. Readiness requires every
    query to qualify and both recall and checks to meet ``max_error`` in output
    units. ``epochs=0`` only assesses readiness. No inputs or targets are scaled.

    Any refusal stops the helper with ``passed=False``. Earlier accepted
    witnesses remain committed; the whole bootstrap call is not one transaction.
    Each minibatch is atomic. The report
    retains work, presentation counts, validation history and failure details.
    Replays are counted as presentations, not newly collected experiences.
    The same brain continues into the live phase without a mode change.

    ``batch_size=1`` uses ordered ``observe`` calls. Larger positive sizes group
    each shuffled epoch into atomic ``observe_batch`` calls (including a shorter
    last batch), preserving live activity. ``presentations`` and ``accepted``
    count examples; ``updates`` counts committed calls. Batching minimizes mean
    example energy with one parameter anchor per batch, so it changes the
    learning trajectory rather than emulating sequential admissions.
    """
    if not isinstance(brain, Brain):
        raise ValueError("brain must be a Brain")
    epochs = integer(epochs, "epochs")
    seed = integer(seed, "seed")
    budget = None if budget is None else integer(budget, "budget")
    batch_size = integer(batch_size, "batch_size", 1)
    max_error = number(max_error, "max_error")
    if max_error < 0:
        raise ValueError("max_error must be nonnegative")
    examples = _pairs(brain, examples, "examples")
    checks = _pairs(brain, checks, "checks")
    report = {
        "options": {
            "max_error": max_error,
            "epochs": epochs,
            "seed": seed,
            "budget": brain.config["settle_budget"] if budget is None else budget,
            "batch_size": batch_size,
        },
        "passed": False,
        "reason": "epochs",
        "epochs": 0,
        "presentations": 0,
        "accepted": 0,
        "updates": 0,
        "examples": len(examples),
        "checks": len(checks),
        "history": [],
        "work": {},
        "failure": None,
    }
    work = Counter()

    def finish(reason):
        report["reason"] = reason
        report["passed"] = reason == "passed"
        report["work"] = dict(work)
        return report

    def refusal(stage, index, result):
        report["failure"] = {
            "stage": stage,
            "index": index,
            "reason": result["reason"],
            "stationarity": result["stationarity"],
        }

    def evaluate(records, stage):
        metric = {"evaluated": 0, "qualified": 0, "max_error": 0.0}
        for index, (inputs, _, clamps) in enumerate(records):
            result = brain.settle(inputs, budget=budget)
            work.update(result["work"])
            metric["evaluated"] += 1
            if not result["qualified"]:
                metric["max_error"] = None
                refusal(stage, index, result)
                break
            metric["qualified"] += 1
            metric["max_error"] = max(
                metric["max_error"],
                max(abs(result["state"][i] - target) for i, target in clamps.items()),
            )
        return metric

    def ready(epoch):
        entry = {"epoch": epoch, "recall": None, "checks": None}
        report["history"].append(entry)
        for stage, records in (("recall", examples), ("checks", checks)):
            entry[stage] = evaluate(records, stage)
            if report["failure"] is not None:
                return False
        return all(
            entry[stage]["max_error"] <= max_error for stage in ("recall", "checks")
        )

    if ready(0):
        return finish("passed")
    if report["failure"] is not None:
        return finish("refused")
    rng = random.Random(seed)
    for epoch in range(1, epochs + 1):
        order = list(range(len(examples)))
        rng.shuffle(order)
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            if batch_size == 1:
                inputs, targets, _ = examples[indices[0]]
                result = brain.observe(inputs, targets, budget=budget)
            else:
                result = brain.observe_batch(
                    [(examples[i][0], examples[i][1]) for i in indices], budget=budget
                )
            work.update(result["work"])
            report["presentations"] += len(indices)
            if not result["accepted"]:
                refusal(
                    "observe" if batch_size == 1 else "observe_batch",
                    indices[0],
                    result,
                )
                if batch_size > 1:
                    report["failure"]["indices"] = tuple(indices)
                return finish("refused")
            report["accepted"] += len(indices)
            report["updates"] += 1
        report["epochs"] = epoch
        if ready(epoch):
            return finish("passed")
        if report["failure"] is not None:
            return finish("refused")
    return finish("epochs")
```

## 逐行讲解 / What's happening

1. **第 52-81 行 / Lines 52-81**:
   - 中文: docstring 先定义边界：examples 会写入，checks 只做 readiness；batch 是原子提交，但整体调用不是事务。
   - English: The docstring defines the contract: examples are admitted, checks are only readiness probes, batches are atomic, and the whole call is not one transaction.
1. **第 82-119 行 / Lines 82-119**:
   - 中文: 参数全部规范化后，report 先建好默认失败状态；`finish` 只改 reason/passed/work，出口统一。
   - English: After validation, the report starts in a default failure state. `finish` centralizes the reason, passed flag, and work accounting.
1. **第 121-155 行 / Lines 121-155**:
   - 中文: `evaluate` 调 `brain.settle`，不会提交经验；`ready` 同时跑 recall 和 checks，两个都低于 `max_error` 才通过。
   - English: `evaluate` calls `brain.settle` without admitting experience. `ready` runs both recall and checks, and both must meet `max_error`.
1. **第 157-192 行 / Lines 157-192**:
   - 中文: 训练先做 epoch 0 readiness；没过才 shuffle examples 并 observe。任何 refusal 都带着 failure 详情返回。
   - English: Training starts with epoch-0 readiness. Only if that fails does it shuffle examples and observe them. Any refusal returns with failure details.

## 类比 / The analogy

像教练带队训练：先做摸底测试，没达标才安排训练；每轮训练后再测一次。队员受伤或器材不合格，就记录原因并停下。

It is like a coach running practice. Test first, train only if needed, then test again after each round. If a player is injured or equipment fails, record why and stop.

## 自己跑一遍 / Try it yourself

```python
def bootstrap(value, examples, checks, max_error, epochs=3):
    history = []
    def error(records): return max(abs(value - target) for target in records)
    for epoch in range(epochs + 1):
        rec, chk = error(examples), error(checks)
        history.append((epoch, round(rec, 3), round(chk, 3)))
        if rec <= max_error and chk <= max_error:
            return True, history
        if epoch < epochs:
            value += (sum(examples) / len(examples) - value) * 0.5
    return False, history

print(bootstrap(0.0, [1.0], [0.8], 0.25))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
(True, [(0, 1.0, 0.8), (1, 0.5, 0.3), (2, 0.25, 0.05)])
```

这个简化版展示了 readiness-first 的节奏：每一轮都先记录 recall/checks，再决定是否继续。

The simplified version shows the readiness-first rhythm: record recall and checks each round, then decide whether to continue.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **RL replay admission** / **RL replay admission**: 先验证 transition，再写入 replay buffer，避免污染训练集。 / Validate transitions before adding them to replay buffers.
- **CI deployment gates** / **CI deployment gates**: 先跑 smoke/readiness，再允许发布；失败报告要能定位阶段。 / Run smoke/readiness checks before deployment, and keep failure stage details.

## 注意事项 / Caveats / when it breaks

- **checks 不是最终测试集** / **Checks are not a final test set**: docstring 明说 checks 被反复用于 readiness，是开发集。 / The docstring states checks are repeatedly used for readiness, so they are a development set.
- **整体不是事务** / **The full call is not transactional**: 早期已接受 witness 会留下；失败不会自动回滚整次 bootstrap。 / Earlier accepted witnesses remain committed; a later failure does not roll back the whole bootstrap.

## 延伸阅读 / Further reading

- [Cadence repository](https://github.com/muellerberndt/cadence)
- [Bootstrapping guide](https://github.com/muellerberndt/cadence/blob/main/docs/BOOTSTRAP.md)
