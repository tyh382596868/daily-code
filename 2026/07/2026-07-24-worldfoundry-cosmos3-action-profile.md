---
date: 2026-07-24
topic: infrastructure
source: trending
repo: OpenEnvision/WorldFoundry
file: worldfoundry/core/inference.py
permalink: https://github.com/OpenEnvision/WorldFoundry/blob/bc062d7ac08bd911528f6fe7587469f5a6ba21fa/worldfoundry/core/inference.py#L1548-L1687
difficulty: intermediate
read_time: ~9 min
tags: [code-of-the-day, infrastructure, worldfoundry, inference-contract, cosmos3]
---

# WorldFoundry Cosmos3：把动作推理写成任务契约 / WorldFoundry Cosmos3: Turn Action Inference into a Task Contract

> **一句话 / In one line**: `_cosmos3_action_task_profile` 把 action-conditioned world-model 推理的输入字段、默认参数和输出 artifacts 集中声明成一个 profile。 / `_cosmos3_action_task_profile` declares the inputs, defaults, and output artifacts for action-conditioned world-model inference in one profile.

## 为什么重要 / Why this matters

世界模型平台通常同时服务 Studio、CLI、benchmark 和脚本。每个入口如果都手写 `num_frames`、`action_chunk_size`、`view_point`，参数很快会分叉。WorldFoundry 的做法是把一次任务变成 schema：谁需要 UI、命令行或评测，都读同一个 `InferenceTaskProfile`。

World-model platforms usually serve Studio, CLI, benchmarks, and scripts at once. If every entry point hard-codes `num_frames`, `action_chunk_size`, and `view_point`, defaults diverge quickly. WorldFoundry turns each task into a schema: UI, command-line, and evaluation all read the same `InferenceTaskProfile`.

## 代码 / The code

`OpenEnvision/WorldFoundry` — [`worldfoundry/core/inference.py`](https://github.com/OpenEnvision/WorldFoundry/blob/bc062d7ac08bd911528f6fe7587469f5a6ba21fa/worldfoundry/core/inference.py#L1548-L1687)

```python
def _cosmos3_action_task_profile(mode: str, label: str) -> InferenceTaskProfile:
    """Build an official-default Cosmos3 action inference task."""

    task_id = f"action-{mode.replace('_', '-')}"
    defaults: dict[str, Any] = {
        "task_type": task_id,
        "action_mode": mode,
        "action_chunk_size": 16,
        "domain_name": "bridge_orig_lerobot",
        "resolution_tier": 480,
        "view_point": "ego_view",
        "num_frames": 17,
        "fps": 5,
        "num_inference_steps": 30,
        "guidance_scale": 1.0,
        "flow_shift": 10.0,
        "use_karras_sigmas": False,
        "use_system_prompt": False,
        "enable_safety_check": True,
        "enable_sound": False,
        "output_type": "video",
        "seed": 0,
    }
    fields: list[InferenceFieldSpec] = [
        _field("prompt", "Prompt", target="prompt", required=True, default=COSMOS3_DEFAULT_PROMPT),
        _field(
            "load_sound_tokenizer",
            "Load Sound Tokenizer",
            kind="boolean",
            target="load_kwargs",
            default=False,
            description="Action inference is visual/action-only and does not require the AVAE sound tokenizer.",
        ),
        _field(
            "input_path",
            "Input Video",
            kind="path",
            target="input_path",
            required=True,
            default="",
            description=(
                "Policy and forward dynamics use the first frame; inverse dynamics conditions on the clip."
            ),
        ),
        _field(
            "action_mode",
            "Action Mode",
            target="call_kwargs",
            required=True,
            default=mode,
            choices=(mode,),
        ),
        _field(
            "action_chunk_size",
            "Action Chunk Size",
            kind="integer",
            target="call_kwargs",
            required=True,
            default=16,
            description="The generated clip contains chunk_size + 1 frames.",
        ),
    ]
    if mode == "forward_dynamics":
        fields.append(
            _field(
                "raw_actions",
                "Raw Actions",
                kind="json",
                target="call_kwargs",
                required=True,
                description="A [T, D] action array driving the forward-dynamics rollout.",
            )
        )
    outputs = [_artifact("video", "video", required=True, preview=True)]
    if mode in {"policy", "inverse_dynamics"}:
        outputs.append(
            _artifact(
                "action_trace",
                "action_trace",
                description="Predicted normalized actions emitted by the policy or inverse-dynamics head.",
            )
        )
    outputs.append(_artifact("manifest", "manifest", required=True))
    return InferenceTaskProfile(
        task_id=task_id,
        label=label,
        description=f"Run Cosmos3 {label.lower()} with the official action-inference defaults.",
        aliases=tuple(dict.fromkeys((mode, mode.replace("_", "-")))),
        inputs=tuple(fields),
        outputs=tuple(outputs),
        default_call_kwargs=defaults,
    )
```

## 逐行讲解 / What's happening

1. **第 1551 行 / Line 1551 (`task_id`)**: 中文: `forward_dynamics` 这类内部 mode 被转换成稳定的 catalog id。 English: an internal mode like `forward_dynamics` becomes a stable catalog id.
2. **第 1552-1570 行 / Lines 1552-1570 (`defaults`)**: 中文: 推理默认值集中在一张表里，避免 UI 和 CLI 各写一份。 English: inference defaults live in one table instead of being duplicated by UI and CLI.
3. **第 1572-1623 行 / Lines 1572-1623 (`fields`)**: 中文: 每个输入字段都声明类型、目标位置、默认值和可选项。 English: every input field declares type, target, default, and choices.
4. **第 1625-1636 行 / Lines 1625-1636 (`raw_actions`)**: 中文: 只有 forward dynamics 需要用户传动作数组，所以字段按 mode 动态追加。 English: only forward dynamics needs a user action array, so that field is added by mode.
5. **第 1664-1676 行 / Lines 1664-1676 (`outputs`)**: 中文: policy/inverse dynamics 会额外产出 action trace，普通视频任务只产出 video 和 manifest。 English: policy and inverse dynamics emit an extra action trace; plain video tasks emit video and manifest.

## 类比 / The analogy

像餐厅把每道菜写成标准工单：原料、火候、出餐盘和是否需要配酱都在一张单上。厨房、收银和外卖系统不用各自猜规则。

It is like a restaurant turning each dish into a standard ticket: ingredients, heat, plate, and sauce are all on one sheet. Kitchen, cashier, and delivery systems do not guess separate rules.

## 自己跑一遍 / Try it yourself

```python
def action_profile(mode):
    defaults = {"action_mode": mode, "action_chunk_size": 16, "num_frames": 17}
    fields = ["prompt", "input_path", "action_mode", "action_chunk_size"]
    if mode == "forward_dynamics":
        fields.append("raw_actions")
    outputs = ["video", "manifest"]
    if mode in {"policy", "inverse_dynamics"}:
        outputs.append("action_trace")
    return {"task_id": f"action-{mode.replace('_', '-')}", "fields": fields, "outputs": outputs, "defaults": defaults}

print(action_profile("policy"))
print(action_profile("forward_dynamics"))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
{'task_id': 'action-policy', 'fields': ['prompt', 'input_path', 'action_mode', 'action_chunk_size'], 'outputs': ['video', 'manifest', 'action_trace'], 'defaults': {'action_mode': 'policy', 'action_chunk_size': 16, 'num_frames': 17}}
{'task_id': 'action-forward-dynamics', 'fields': ['prompt', 'input_path', 'action_mode', 'action_chunk_size', 'raw_actions'], 'outputs': ['video', 'manifest'], 'defaults': {'action_mode': 'forward_dynamics', 'action_chunk_size': 16, 'num_frames': 17}}
```

中文: mode 改变的不是一堆散落的 if，而是字段和 artifact 契约。 English: the mode changes the field and artifact contract, not scattered conditionals.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **vLLM engine args** / **vLLM engine args**: serving 系统把 runtime 参数集中成配置对象，多个入口共享。 / serving systems centralize runtime knobs into a shared config object.
- **LeRobot policy configs** / **LeRobot policy configs**: policy、dataset、processor 都围绕显式 schema 连接。 / policy, dataset, and processor wiring revolves around explicit schemas.

## 注意事项 / Caveats / when it breaks

- **schema 不是执行器 / A schema is not the runner**: profile 只声明契约，真正模型加载和推理仍在 runner 里。 / the profile declares the contract; the runner still loads and executes the model.
- **默认值会变成产品行为 / Defaults become product behavior**: `seed`、`fps`、`safety_check` 这类值需要被版本化管理。 / values like `seed`, `fps`, and `safety_check` need versioned ownership.
- **mode 分支要少 / Keep mode branches small**: 如果每个 mode 都追加大量专有逻辑，应拆成多个 profile builder。 / if every mode adds a lot of custom logic, split into separate profile builders.

## 延伸阅读 / Further reading

- [WorldFoundry inference.py](https://github.com/OpenEnvision/WorldFoundry/blob/bc062d7ac08bd911528f6fe7587469f5a6ba21fa/worldfoundry/core/inference.py#L1548-L1687)

