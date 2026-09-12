---
date: 2026-07-30
topic: wam
source: wam
repo: huggingface/lerobot
file: src/lerobot/policies/fastwam/processor_fastwam.py
permalink: https://github.com/huggingface/lerobot/blob/36b8face988669509272b00f4abe6592d0b17aa0/src/lerobot/policies/fastwam/processor_fastwam.py#L35-L115
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, wam, action-conditioning, postprocess]
build_role: action-conditioning advanced variant
---

# FastWAM action toggle：把连续输出接到 LIBERO 开关语义 / FastWAM Action Toggle: Attach Continuous Outputs to LIBERO Toggle Semantics

> **一句话 / In one line**: `FastWAMActionToggleProcessorStep` 在输出 pipeline 末端把指定动作维从 `[0,1]` 风格值翻译成 toggle 符号。 / `FastWAMActionToggleProcessorStep` converts selected action dimensions from `[0,1]`-style values into toggle signs at the end of the output pipeline.

## 为什么重要 / Why this matters

WAM 采样器吐出的动作 latent 不等于环境能吃的命令。FastWAM 在 processor pipeline 末端保留一个很小的后处理层，专门处理 LIBERO 这种“开/关”动作维度。

The action latent emitted by a WAM sampler is not automatically an environment command. FastWAM keeps a tiny postprocessing step at the end of the processor pipeline specifically for LIBERO-style on/off action dimensions.

## 代码 / The code

`huggingface/lerobot` — [`src/lerobot/policies/fastwam/processor_fastwam.py`](https://github.com/huggingface/lerobot/blob/36b8face988669509272b00f4abe6592d0b17aa0/src/lerobot/policies/fastwam/processor_fastwam.py#L35-L115)

```python
@dataclass
@ProcessorStepRegistry.register(name="fastwam_action_toggle_processor")
class FastWAMActionToggleProcessorStep(ActionProcessorStep):
    """Apply FastWAM LIBERO toggle semantics to configured action dimensions."""

    toggle_dimensions: list[int]

    def action(self, action: PolicyAction) -> PolicyAction:
        if not self.toggle_dimensions:
            return action
        processed_action = action.clone()
        action_dim = int(processed_action.shape[-1])
        for dim in self.toggle_dimensions:
            resolved_dim = dim if dim >= 0 else action_dim + dim
            if resolved_dim < 0 or resolved_dim >= action_dim:
                raise ValueError(
                    f"FastWAM action toggle dimension {dim} is out of bounds for action dim {action_dim}."
                )
            value = processed_action[..., resolved_dim]
            value = value * 2.0 - 1.0
            processed_action[..., resolved_dim] = torch.sign(-value)
        return processed_action

    def get_config(self) -> dict[str, Any]:
        return {"toggle_dimensions": self.toggle_dimensions}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def make_fastwam_pre_post_processors(
    config: FastWAMConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[PolicyProcessorPipeline, PolicyProcessorPipeline]:
    """Create LeRobot pre- and post-processing pipelines for FastWAM.

    Args:
        config (FastWAMConfig): Policy configuration controlling device and
            normalization feature metadata.
        dataset_stats (dict[str, dict[str, torch.Tensor]] | None): Optional
            LeRobot dataset statistics used by normalization processors.

    Returns:
        tuple[PolicyProcessorPipeline, PolicyProcessorPipeline]: Input and
        output processor pipelines discoverable by LeRobot.
    """

    # NOTE: no visual normalization here. VISUAL is IDENTITY (see configuration_fastwam.normalization_mapping)
    # — images pass through in [0, 1] and the model maps them to the Wan VAE's [-1, 1] at the encode
    # boundary. This is deliberate: `lerobot_train.py` overrides the normalizer stats with
    # `dataset.meta.stats` when fine-tuning, and a real dataset's per-channel image std is the tiny
    # frame-to-frame brightness variance, which would blow images far outside [-1,1] and saturate them.
    # STATE/ACTION still normalize with dataset stats below.
    normalization_stats: dict[str, dict[str, Any]] = dict(dataset_stats or {})

    # NOTE: no resize step here. The model is the single authority on input resolution: it resizes
    # each camera to the per-camera target (image_size split across cameras) in
    # `_stack_video_from_images` / `_prepare_infer_image`, on every path (train forward, rollout and
    # eval select_action). A preprocessor resize step would be both redundant (the model re-resizes
    # anyway) and unsafe across fine-tuning: its `resize_size` would be inherited from the base
    # checkpoint's camera geometry, not this dataset's, making the concatenation N_cameras x too wide.

    steps = make_default_policy_processor_steps(config, normalization_stats, normalizer_device=config.device)

    input_steps = [
        steps.rename_observations,
        steps.add_batch_dim,
        steps.to_device,
        steps.normalize,
    ]
    output_steps = [
        steps.unnormalize,
    ]
    if config.toggle_action_dimensions:
        output_steps.append(
            FastWAMActionToggleProcessorStep(toggle_dimensions=config.toggle_action_dimensions)
        )
    output_steps.append(steps.to_cpu)
    return make_policy_processor_pipelines(input_steps=input_steps, output_steps=output_steps)
```

## 逐行讲解 / What's happening

1. **第 35-42 行 / Lines 35-42 (registered dataclass)**:
   - 中文: toggle processor 被注册成 pipeline step，配置里只需要给维度列表。
   - English: The toggle processor is registered as a pipeline step; configs only need a list of dimensions.
2. **第 43-56 行 / Lines 43-56 (`action`)**:
   - 中文: 先 clone 输出，再逐维解析负索引、检查边界、做 `value * 2 - 1` 和 `sign(-value)`。
   - English: It clones the output, resolves negative indices, checks bounds, then applies `value * 2 - 1` and `sign(-value)`.
3. **第 67-99 行 / Lines 67-99 (pipeline comments)**:
   - 中文: 视觉不在 processor 里 normalize/resize；这些边界交给模型内部处理。
   - English: Visual normalization/resizing is not done in the processor; those boundaries are owned by the model.
4. **第 101-115 行 / Lines 101-115 (input/output steps)**:
   - 中文: 输入只做 rename/batch/device/normalize；输出先 unnormalize，再可选 toggle，最后回 CPU。
   - English: Inputs get rename/batch/device/normalize; outputs get unnormalize, optional toggle, then CPU transfer.

## 类比 / The analogy

这像门禁系统：模型给的是“旋钮位置”，真实门锁需要的是“开门/关门”的离散方向，最后要有一个小转换器。

It is like a door access system: the model gives a knob position, but the lock needs an open/close direction, so a small final converter is required.

## 在 nanoVLA / nanoWAM 中的位置 / Where this lives in your nano-{VLA,WAM}

在 nanoWAM 里，这可以放在 `action-conditioning` / action-output boundary 的 advanced variant：采样器负责生成连续动作轨迹，processor 负责把轨迹适配到具体环境的动作语义。省掉它，模型输出和控制接口之间会夹着隐式约定。

In a nanoWAM, this fits as an advanced variant at the `action-conditioning` / action-output boundary: the sampler generates a continuous action trajectory, and the processor adapts it to the environment action semantics. Without it, model output and control interface are connected by implicit assumptions.

## 自己跑一遍 / Try it yourself

```python
def toggle(action, dims):
    out = list(action)
    n = len(out)
    for dim in dims:
        i = dim if dim >= 0 else n + dim
        if i < 0 or i >= n:
            raise ValueError('toggle dimension out of bounds')
        value = out[i] * 2.0 - 1.0
        out[i] = -1 if value > 0 else (1 if value < 0 else 0)
    return out

print(toggle([0.2, 0.8, 0.5], [0, 1, -1]))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[1, -1, 0]
```

这个例子展示了两个工程点：支持负索引，并且 toggle 后输出只剩符号语义。

This example shows two engineering details: negative indices are supported, and the output becomes sign semantics after toggling.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **LeRobot action spaces** / **LeRobot action spaces**: action head 后面也会有语义化 postprocess。 / Action heads also often have semantic postprocessing.
- **Gym action wrappers** / **Gym action wrappers**: 连续 policy 输出常通过 wrapper 映射到环境 action space。 / Continuous policy outputs are often mapped into environment action spaces through wrappers.

## 注意事项 / Caveats / when it breaks

- **维度配置错会硬失败** / **Wrong dimensions fail hard**: 代码主动检查越界，避免悄悄改错维度。 / The code checks bounds to avoid silently changing the wrong dimension.
- **环境语义绑定** / **Environment-specific semantics**: 这个 processor 很实用，但绑定 LIBERO toggle 约定，换环境要重审。 / This processor is useful but tied to LIBERO toggle semantics; changing environments requires review.

## 延伸阅读 / Further reading

- [FastWAM in LeRobot](https://github.com/huggingface/lerobot)
- [Source permalink](https://github.com/huggingface/lerobot/blob/36b8face988669509272b00f4abe6592d0b17aa0/src/lerobot/policies/fastwam/processor_fastwam.py#L35-L115)
