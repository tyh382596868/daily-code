---
date: 2026-08-13
topic: robotics
source: tracked
repo: huggingface/lerobot
file: src/lerobot/processor/relative_action_processor.py
permalink: https://github.com/huggingface/lerobot/blob/a16f34c085c9597fcbdb9fde395a3334d78df716/src/lerobot/processor/relative_action_processor.py#L40-L143
difficulty: beginner
read_time: ~10 min
tags: [code-of-the-day, robotics, relative-actions]
---

# LeRobot relative actions：把绝对目标改成相对位移 / LeRobot Relative Actions: Turn Absolute Targets into Offsets

> **一句话 / In one line**: `RelativeActionsProcessorStep` 用当前 `observation.state` 做参考点，把需要相对控制的动作维度改写成 `action - state`。 / `RelativeActionsProcessorStep` uses the current `observation.state` as a reference and rewrites selected action dimensions as `action - state`.

## 为什么重要 / Why this matters

机器人策略常常更容易学习“往左移 2cm”而不是“移动到世界坐标 0.431”。这段代码把这种建模选择放进 processor，而不是塞进模型本体：训练时模型看相对动作，执行时再还原成机器人需要的绝对目标。

Robot policies often learn “move 2 cm left” more easily than “go to world coordinate 0.431”. This code keeps that modeling choice in a processor rather than inside the network: training sees relative actions, while execution can recover absolute robot targets.

## 代码 / The code

`huggingface/lerobot` — [`src/lerobot/processor/relative_action_processor.py`](https://github.com/huggingface/lerobot/blob/a16f34c085c9597fcbdb9fde395a3334d78df716/src/lerobot/processor/relative_action_processor.py#L40-L143)

```python
def to_relative_actions(actions: Tensor, state: Tensor, mask: Sequence[bool]) -> Tensor:
    """Convert absolute actions to relative: relative = action - state (for masked dims).

    Args:
        actions: (B, T, action_dim) or (B, action_dim).
        state: (B, state_dim). Broadcast across time dimension.
        mask: Which dims to convert. Can be shorter than action_dim.
    """
    mask_t = torch.tensor(mask, dtype=actions.dtype, device=actions.device)
    dims = mask_t.shape[0]
    # Align state to the same device/dtype as actions. _last_state is cached before
    # DeviceProcessorStep moves the transition, so it can be on CPU while actions are on CUDA.
    if state.device != actions.device or state.dtype != actions.dtype:
        state = state.to(device=actions.device, dtype=actions.dtype)
    state_offset = state[..., :dims] * mask_t
    if actions.ndim == 3:
        state_offset = state_offset.unsqueeze(-2)
    actions = actions.clone()
    actions[..., :dims] -= state_offset
    return actions


def to_absolute_actions(actions: Tensor, state: Tensor, mask: Sequence[bool]) -> Tensor:
    """Convert relative actions back to absolute: absolute = relative + state (for masked dims).

    Args:
        actions: (B, T, action_dim) or (B, action_dim).
        state: (B, state_dim). Broadcast across time dimension.
        mask: Which dims to convert. Can be shorter than action_dim.
    """
    mask_t = torch.tensor(mask, dtype=actions.dtype, device=actions.device)
    dims = mask_t.shape[0]
    # Align state to the same device/dtype as actions. _last_state is cached before
    # DeviceProcessorStep moves the transition, so it can be on CPU while actions are on CUDA.
    if state.device != actions.device or state.dtype != actions.dtype:
        state = state.to(device=actions.device, dtype=actions.dtype)
    state_offset = state[..., :dims] * mask_t
    if actions.ndim == 3:
        state_offset = state_offset.unsqueeze(-2)
    actions = actions.clone()
    actions[..., :dims] += state_offset
    return actions


@ProcessorStepRegistry.register("relative_actions_processor")
@dataclass
class RelativeActionsProcessorStep(ProcessorStep):
    """Converts absolute actions to relative actions (action -= state) for masked dimensions.

    Mirrors OpenPI's DeltaActions transform. Applied during preprocessing so the model
    trains on relative offsets instead of absolute positions.
    Caches the last seen state so a paired AbsoluteActionsProcessorStep can reverse
    the conversion during postprocessing.

    Attributes:
        enabled: Whether to apply the relative conversion.
        exclude_joints: Joint names to keep absolute (not converted to relative).
        action_names: Action dimension names from dataset metadata, used to build
            the mask from exclude_joints. If None, all dims are converted.
    """

    enabled: bool = False
    exclude_joints: list[str] = field(default_factory=list)
    action_names: list[str] | None = None
    _last_state: torch.Tensor | None = field(default=None, init=False, repr=False)

    def _build_mask(self, action_dim: int) -> list[bool]:
        if not self.exclude_joints or self.action_names is None:
            return [True] * action_dim

        exclude_tokens = [str(name).lower() for name in self.exclude_joints if name]
        if not exclude_tokens:
            return [True] * action_dim

        mask = []
        for name in self.action_names[:action_dim]:
            action_name = str(name).lower()
            is_excluded = any(token == action_name or token in action_name for token in exclude_tokens)
            mask.append(not is_excluded)

        if len(mask) < action_dim:
            mask.extend([True] * (action_dim - len(mask)))

        return mask

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION, {})
        state = observation.get(OBS_STATE) if observation else None

        # Always cache state for the paired AbsoluteActionsProcessorStep
        if state is not None:
            self._last_state = state

        if not self.enabled:
            return transition

        new_transition = transition.copy()
        action = new_transition.get(TransitionKey.ACTION)
        if action is None or state is None:
            return new_transition

        mask = self._build_mask(action.shape[-1])
        new_transition[TransitionKey.ACTION] = to_relative_actions(action, state, mask)
        return new_transition
```

## 逐行讲解 / What's happening

1. **第 48-59 行 / Lines 48-59: mask 会搬到 action 的 dtype/device，只对被选维度减去 state；`clone()` 避免改坏上游 transition。 / The mask is moved to the action dtype/device, only selected dimensions subtract state, and `clone()` avoids mutating the upstream transition.**
2. **第 70-81 行 / Lines 70-81: absolute 路径是对称加法，所以同一份 state 可以把相对动作还原。 / The absolute path is symmetric addition, so the same state can recover the relative action.**
3. **第 106-123 行 / Lines 106-123: `exclude_joints` 构造语义 mask，让 gripper 等维度保持绝对值。 / `exclude_joints` builds the semantic mask, keeping gripper-like dimensions absolute.**
4. **第 125-143 行 / Lines 125-143: processor 总是缓存 state，但只有 `enabled` 时才改写 action。 / The processor always caches state, but rewrites actions only when `enabled` is true.**

## 类比 / The analogy

像导航时说“从当前位置往北走两格”，而不是每次都报一个全球坐标。当前位置可以变，但偏移动作更稳定。

It is like saying “move two blocks north from here” instead of giving a global coordinate every time. The current position may change, but the offset command stays stable.

## 自己跑一遍 / Try it yourself

```python
def to_relative(actions, state, mask):
    out = [row[:] for row in actions]
    for row in out:
        for i, use in enumerate(mask):
            if use:
                row[i] -= state[i]
    return out

def to_absolute(actions, state, mask):
    out = [row[:] for row in actions]
    for row in out:
        for i, use in enumerate(mask):
            if use:
                row[i] += state[i]
    return out

state = [10.0, 20.0, 0.5]
absolute = [[11.0, 18.0, 0.1], [12.0, 21.0, 0.0]]
mask = [True, True, False]
relative = to_relative(absolute, state, mask)
print(relative)
print(to_absolute(relative, state, mask))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[[1.0, -2.0, 0.1], [2.0, 1.0, 0.0]]
[[11.0, 18.0, 0.1], [12.0, 21.0, 0.0]]
```

第三维被 mask 排除，所以 gripper 一类维度不会被 state 偏移污染。

The third dimension is excluded by the mask, so gripper-like dimensions are not polluted by state offsets.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **openpi DeltaActions** / **openpi DeltaActions**: 同样把绝对动作变成相对 state 的增量。 / It also turns absolute actions into state-relative offsets.
- **Isaac-GR00T action chunks** / **Isaac-GR00T action chunks**: 长 horizon 控制常混用相对位移、逐步 delta 和绝对目标。 / Long-horizon control often mixes offsets, per-step deltas, and absolute targets.

## 注意事项 / Caveats / when it breaks

- **state 必须配对** / **State must be paired**: 前处理和后处理要看到同一帧 state。 / Pre- and post-processing must see the same state frame.
- **mask 是语义合同** / **The mask is semantic**: `exclude_joints` 配错会让部分维度在错误坐标系里学习。 / A wrong `exclude_joints` setting trains some dimensions in the wrong frame.

## 延伸阅读 / Further reading

- [huggingface/lerobot source](https://github.com/huggingface/lerobot/blob/a16f34c085c9597fcbdb9fde395a3334d78df716/src/lerobot/processor/relative_action_processor.py#L40-L143)
