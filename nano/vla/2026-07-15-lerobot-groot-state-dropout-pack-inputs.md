---
date: 2026-07-15
topic: vla
source: vla
repo: huggingface/lerobot
file: src/lerobot/policies/groot/processor_groot.py
permalink: https://github.com/huggingface/lerobot/blob/e40b58a8dfa9e7b86918c374791599d070518d11/src/lerobot/policies/groot/processor_groot.py#L1485-L1948
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, vla, groot, preprocessing, action-mask]
build_role: training-step advanced variant
---

# GR00T N1.7 PackInputs：训练时打包 state、action 和 mask / GR00T N1.7 PackInputs: Pack State, Action, and Masks for Training

> **一句话 / In one line**: LeRobot 的 GR00T N1.7 preprocessor 把原始 transition 变成固定形状张量，并在训练时对 state 做整样本 dropout。 / LeRobot's GR00T N1.7 preprocessor turns raw transitions into fixed-shape tensors and applies whole-sample state dropout during training.

## 为什么重要 / Why this matters

VLA 训练常败在“模型结构正确，但输入契约不一致”。GR00T N1.7 要固定最大 state/action 维度、固定 action horizon、相机顺序、embodiment id、action mask，还要在训练时复现 checkpoint 侧车配置里的 state dropout。这个 step 就是把环境数据整理成模型真正吃的格式。

VLA training often fails when the architecture is right but the input contract is not. GR00T N1.7 expects fixed maximum state/action dimensions, a fixed action horizon, ordered cameras, an embodiment id, an action mask, and training-time state dropout from checkpoint sidecar settings. This step is where environment data becomes the exact format the model consumes.

## 代码 / The code

`huggingface/lerobot` — [`src/lerobot/policies/groot/processor_groot.py`](https://github.com/huggingface/lerobot/blob/e40b58a8dfa9e7b86918c374791599d070518d11/src/lerobot/policies/groot/processor_groot.py#L1485-L1948)

```python
@dataclass
@ProcessorStepRegistry.register(name="groot_n1_7_pack_inputs_v1")
class GrootN17PackInputsStep(ProcessorStep):
    """Pack LeRobot transitions into the raw tensor layout expected by N1.7."""

    state_horizon: int = 1
    action_horizon: int = N1_7_NATIVE_ACTION_HORIZON
    valid_action_horizon: int = N1_7_NATIVE_ACTION_HORIZON
    max_state_dim: int = 132
    max_action_dim: int = 132
    normalize_min_max: bool = True
    training: bool = False
    state_dropout_prob: float = 0.0
    stats: dict[str, dict[str, Any]] | None = None

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        obs = transition.get(TransitionKey.OBSERVATION, {}) or {}
        comp = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}) or {}
        raw_state_for_action: torch.Tensor | None = None

        if OBS_STATE in obs:
            state = obs[OBS_STATE]
            if state.dim() != 2:
                raise ValueError(f"state must be (B, D), got {tuple(state.shape)}")
            bsz, dim = state.shape
            if dim > self.max_state_dim:
                raise ValueError(f"State dimension {dim} exceeds max_state_dim {self.max_state_dim}.")
            raw_state_for_action = state
            if self.normalize_min_max:
                state = _min_max_norm(state, OBS_STATE)
            state = state.unsqueeze(1)
            if dim < self.max_state_dim:
                pad = torch.zeros(bsz, 1, self.max_state_dim - dim, dtype=state.dtype, device=state.device)
                state = torch.cat([state, pad], dim=2)
            if self.training and torch.is_grad_enabled() and self.state_dropout_prob > 0:
                drop_state = torch.tensor(
                    [random.random() < self.state_dropout_prob for _ in range(bsz)],
                    dtype=torch.bool,
                    device=state.device,
                ).view(bsz, 1, 1)
                state = state.masked_fill(drop_state, 0)
            obs["state"] = state

        action = transition.get(TransitionKey.ACTION)
        if isinstance(action, torch.Tensor):
            if action.dim() == 2:
                action = action.unsqueeze(1)
            elif action.dim() == 3:
                pass
            else:
                raise ValueError(f"action must be (B, D) or (B, T, D), got {tuple(action.shape)}")

            bsz, horizon, dim = action.shape
            if horizon > self.action_horizon:
                raise ValueError(f"Action horizon {horizon} exceeds action_horizon {self.action_horizon}.")
            if dim > self.max_action_dim:
                raise ValueError(f"Action dimension {dim} exceeds max_action_dim {self.max_action_dim}.")
            if raw_state_for_action is not None:
                action = self._convert_relative_action_groups_for_training(action, raw_state_for_action)
            if self.normalize_min_max:
                normalized_action = self._normalize_action_groups_for_training(action)
                if normalized_action is not None:
                    action = normalized_action
                else:
                    flat = _min_max_norm(action.reshape(bsz * horizon, dim), ACTION)
                    action = flat.view(bsz, horizon, dim)
            valid_dim = min(dim, self.max_action_dim)
            valid_horizon = min(horizon, self.valid_action_horizon, self.action_horizon)
            if dim < self.max_action_dim:
                pad = torch.zeros(
                    bsz, horizon, self.max_action_dim - dim, dtype=action.dtype, device=action.device
                )
                action = torch.cat([action, pad], dim=2)
            if horizon < self.action_horizon:
                pad = torch.zeros(
                    bsz,
                    self.action_horizon - horizon,
                    self.max_action_dim,
                    dtype=action.dtype,
                    device=action.device,
                )
                action = torch.cat([action, pad], dim=1)
                horizon = self.action_horizon
            horizon_valid = torch.zeros(bsz, horizon, dtype=torch.bool, device=action.device)
            horizon_valid[:, :valid_horizon] = True
            action_mask = torch.zeros(
                bsz, horizon, self.max_action_dim, dtype=torch.float32, device=action.device
            )
            action_mask[:, :, :valid_dim] = horizon_valid.unsqueeze(-1).to(dtype=action_mask.dtype)
            transition[TransitionKey.ACTION] = action
            comp["action_mask"] = action_mask
```

## 逐行讲解 / What's happening

1. **固定上限 / Fixed caps**:
   - 中文: `max_state_dim` / `max_action_dim` 是模型合同，不是当前环境的真实维度；小于上限就补零，大于上限直接报错。
   - English: `max_state_dim` and `max_action_dim` are model-contract sizes, not environment sizes; smaller inputs are padded, larger ones fail fast.
2. **state dropout**:
   - 中文: 只在 `training` 且 `grad_enabled` 时生效，按样本整条 state 置零，逼模型不要过度依赖本体状态。
   - English: It runs only during training with gradients enabled, zeroing whole state samples so the model does not overdepend on proprioception.
3. **relative action conversion**:
   - 中文: 如果 action 以相对量训练，先用 raw state 做 reference，再归一化；顺序不能反。
   - English: For relative-action training, raw state is used as the reference before normalization; reversing the order changes the meaning.
4. **`action_mask`**:
   - 中文: mask 同时编码“哪些 horizon 有效”和“哪些 action 维度有效”，loss 可以直接按 mask 加权。
   - English: The mask encodes both valid timesteps and valid action dimensions, so the loss can weight only real targets.

## 类比 / The analogy

这像把不同学校的成绩单塞进统一申请表：有的学校 5 门课，有的 8 门课，但申请系统只收 10 个格子。空格要补零，还要附一张“哪些格子是真的”的说明表，评审才不会把空格当作成绩。

It is like putting transcripts from different schools into one application form. One school has five courses, another has eight, but the system accepts ten slots. Empty slots are padded, and a separate sheet marks which slots are real so reviewers do not grade blanks.

## 在 nanoVLA 中的位置 / Where this lives in your nano-VLA

中文: 这是 `training-step` 的高级变体，位于 dataset/collator 之后、VLA forward 之前。你的 nanoVLA 也需要一个 `PackInputs`：输入是相机图像、state、action chunk、task language；输出是固定形状的 `state`、`action`、`action_mask` 和 metadata。省掉它，模型会把 padding 当监督，或者在不同 embodiment 间把 action 维度对错。

English: This is an advanced `training-step` component, sitting after the dataset/collator and before the VLA forward pass. A nanoVLA needs its own `PackInputs`: images, state, action chunk, and task language in; fixed-shape `state`, `action`, `action_mask`, and metadata out. Without it, the model trains on padding or misaligns action dimensions across embodiments.

## 自己跑一遍 / Try it yourself

```python
import numpy as np

B, state_dim, max_state_dim = 2, 3, 5
horizon, max_horizon, act_dim, max_act_dim = 2, 4, 2, 4
state = np.array([[1., 2., 3.], [4., 5., 6.]])
action = np.ones((B, horizon, act_dim))

state_padded = np.pad(state[:, None, :], ((0, 0), (0, 0), (0, max_state_dim - state_dim)))
drop = np.array([True, False])[:, None, None]
state_padded = np.where(drop, 0, state_padded)

action_padded = np.pad(action, ((0, 0), (0, max_horizon - horizon), (0, max_act_dim - act_dim)))
mask = np.zeros_like(action_padded)
mask[:, :horizon, :act_dim] = 1
print(state_padded)
print(mask[0])
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[[[0. 0. 0. 0. 0.]]
 [[4. 5. 6. 0. 0.]]]
[[1. 1. 0. 0.]
 [1. 1. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]]
```

中文: 第一个样本 state 被 dropout，action mask 仍然准确标出真实监督区域。

English: The first sample's state is dropped, while the action mask still marks the real supervised region.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **LeRobot ACT / DiffusionPolicy** / **LeRobot ACT / DiffusionPolicy**: 也会把 action chunk 和 mask 整理成固定 horizon。 / They also package action chunks and masks into a fixed horizon.
- **OpenPI transforms** / **OpenPI transforms**: 输入/输出 transforms 把 dataset schema 映射到 policy schema。 / Input/output transforms map dataset schemas into policy schemas.

## 注意事项 / Caveats / when it breaks

- **dropout 只该训练用 / Dropout is training-only**: eval 或 rollout 时打开会让策略失去本体状态。
- **mask 比 padding 更重要 / Mask matters more than padding**: padding 数值是 0 不代表 loss 会自动忽略它。
- **relative action 依赖 raw state / Relative actions depend on raw state**: 先归一化再做 delta 会改变动作单位。

## 延伸阅读 / Further reading

- [LeRobot `processor_groot.py`](https://github.com/huggingface/lerobot/blob/e40b58a8dfa9e7b86918c374791599d070518d11/src/lerobot/policies/groot/processor_groot.py)
- [LeRobot GR00T policy docs](https://github.com/huggingface/lerobot/tree/e40b58a8dfa9e7b86918c374791599d070518d11/src/lerobot/policies/groot)
