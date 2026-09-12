---
date: 2026-07-29
topic: vla
source: vla
repo: huggingface/lerobot
file: src/lerobot/policies/xvla/action_hub.py
permalink: https://github.com/huggingface/lerobot/blob/f37be3edbee60f3a09a5183788b91eb19f0c07d1/src/lerobot/policies/xvla/action_hub.py#L30-L172
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, vla, action-head, registry]
build_role: action-head-continuous advanced variant
---

# XVLA action space：让动作 head 有机器人语义 / XVLA Action Spaces: Give the Action Head Robot Semantics

> **一句话 / In one line**: XVLA 把动作向量的维度、loss、预处理和后处理封装成可注册的 action space。 / XVLA wraps action dimensionality, loss, preprocessing, and postprocessing into registered action spaces.

## 为什么重要 / Why this matters

VLA 模型最后通常只吐出一个连续张量，但不同机器人对这个张量的解释完全不同：末端位姿、关节角、夹爪 logits、padding 维度都可能混在一起。action space registry 把这些语义显式化，避免 action head 变成一团硬编码。

A VLA model often emits one continuous tensor, but different robots interpret that tensor very differently: end-effector pose, joint angles, gripper logits, and padding dimensions can all coexist. An action-space registry makes those semantics explicit instead of burying them in hard-coded action heads.

## 代码 / The code

`huggingface/lerobot` — [`src/lerobot/policies/xvla/action_hub.py`](https://github.com/huggingface/lerobot/blob/f37be3edbee60f3a09a5183788b91eb19f0c07d1/src/lerobot/policies/xvla/action_hub.py#L30-L172)

```python
def register_action(name: str):
    """Decorator for registering a new action space."""

    def _wrap(cls):
        key = name.lower()
        if key in ACTION_REGISTRY:
            raise KeyError(f"ActionSpace '{key}' already registered -> {ACTION_REGISTRY[key]}")
        ACTION_REGISTRY[key] = cls
        cls.name = key
        return cls

    return _wrap


def build_action_space(name: str, **kwargs) -> BaseActionSpace:
    """Instantiate a registered action space by name."""
    key = name.lower()
    if key not in ACTION_REGISTRY:
        raise KeyError(f"Unknown action space '{name}'. Available: {list(ACTION_REGISTRY.keys())}")
    return ACTION_REGISTRY[key](**kwargs)


class BaseActionSpace(nn.Module):
    """
    Abstract base class for all action-space definitions.

    Each subclass defines:
      - `dim_action`: dimension of the action vector.
      - `gripper_idx`: indices of gripper channels.
      - `compute_loss(pred, target)`: supervised loss for this space.
      - `preprocess(proprio, action, mode)`: pre-step modifications.
      - `postprocess(action)`: post-step corrections (e.g. apply sigmoid).
    """

    name: str = "base"
    dim_action: int = 0
    gripper_idx: tuple[int, ...] = ()

    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        """Alias for compute_loss."""
        return self.compute_loss(pred, target)

    def preprocess(self, proprio: torch.Tensor, action: torch.Tensor, mode: str = "train"):
        """Default: return unchanged."""
        return proprio, action

    def postprocess(self, action: torch.Tensor) -> torch.Tensor:
        """Default: return unchanged."""
        return action


@register_action("ee6d")
class EE6DActionSpace(BaseActionSpace):
    """End-effector layout with xyz, 6D rotation, and gripper channels."""

    dim_action = 20
    gripper_idx = (9, 19)
    GRIPPER_SCALE = 1.0
    XYZ_SCALE = 500.0
    ROT_SCALE = 10.0
```

## 逐行讲解 / What's happening

1. **第 30-41 行 / Lines 30-41 (`register_action`)**:
   - 中文: 装饰器把 action space 类挂到全局 registry，并防止同名覆盖。
   - English: The decorator stores an action-space class in the global registry and prevents duplicate names.
2. **第 44-49 行 / Lines 44-49 (`build_action_space`)**:
   - 中文: 配置里只需要写名字，运行时再实例化对应 action space。
   - English: Configs only need a name; runtime instantiates the matching action space.
3. **第 55-98 行 / Lines 55-98 (`BaseActionSpace`)**:
   - 中文: 基类规定 action head 必须回答三个问题：怎么算 loss、训练前怎么改 action、推理后怎么改 action。
   - English: The base class requires three answers from an action head: how to compute loss, how to preprocess actions, and how to postprocess predictions.
4. **第 113-172 行 / Lines 113-172 (`EE6DActionSpace`)**:
   - 中文: 20 维动作里，位置、6D 旋转和夹爪分别有自己的 loss 权重与后处理。
   - English: In the 20D action vector, position, 6D rotation, and gripper channels get separate loss weights and postprocessing.

## 类比 / The analogy

这像给同一把多功能遥控器换贴纸：按键数量没变，但“第 9 个键是夹爪”还是“第 9 个键是关节角”必须由贴纸说明。

It is like swapping labels on the same multi-function remote control: the number of buttons is unchanged, but the label must say whether button 9 means gripper or joint angle.

## 在 nanoVLA / nanoWAM 中的位置 / Where this lives in your nano-{VLA,WAM}

在 nanoVLA 里，这属于 `action-head-continuous` 的 advanced variant。视觉编码器和 VLM backbone 负责给出状态表示，action head 输出连续动作张量，而 action space 负责解释这个张量：哪些维度回归、哪些维度走 BCE、哪些维度推理后要 sigmoid。

In a nanoVLA, this is an advanced variant of `action-head-continuous`. The vision encoder and VLM backbone produce state representations, the action head emits a continuous tensor, and the action space interprets it: which dimensions are regressed, which use BCE, and which need sigmoid after inference.

## 自己跑一遍 / Try it yourself

```python
REGISTRY = {}
def register(name):
    def wrap(cls):
        REGISTRY[name] = cls
        return cls
    return wrap

class BaseAction:
    def preprocess(self, action): return action
    def postprocess(self, action): return action

@register("ee")
class EEAction(BaseAction):
    gripper_idx = 3
    def postprocess(self, action):
        out = list(action)
        out[self.gripper_idx] = 1 / (1 + pow(2.71828, -out[self.gripper_idx]))
        return out

space = REGISTRY["ee"]()
print([round(x, 3) for x in space.postprocess([0.1, 0.2, 0.3, 0.0])])
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[0.1, 0.2, 0.3, 0.5]
```

最值得注意的是：模型输出仍然只是列表，但 action space 给第 4 维附上了“夹爪 logits”的语义。

The important point is that the model output is still just a list, but the action space gives the fourth dimension the semantics of a gripper logit.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **LeRobot GR00T processors** / **LeRobot GR00T processors**: processor pipeline 也把 action schema 和转换逻辑分开。 / Processor pipelines similarly separate action schema from conversion logic.
- **OpenPI transforms** / **OpenPI transforms**: transform graph 负责把 dataset 字段改成模型字段。 / Transform graphs map dataset fields into model fields.

## 注意事项 / Caveats / when it breaks

- **维度约定必须一致** / **Dimension contracts must match**: dataset、normalizer、action head 和 robot controller 必须使用同一套维度语义。 / Dataset, normalizer, action head, and robot controller must share the same dimension semantics.
- **loss 权重会改变行为** / **Loss weights change behavior**: `XYZ_SCALE`、`ROT_SCALE` 等数值会决定训练优先级。 / Values like `XYZ_SCALE` and `ROT_SCALE` change training priorities.

## 延伸阅读 / Further reading

- [LeRobot repository](https://github.com/huggingface/lerobot)
- [Source permalink](https://github.com/huggingface/lerobot/blob/f37be3edbee60f3a09a5183788b91eb19f0c07d1/src/lerobot/policies/xvla/action_hub.py#L30-L172)
