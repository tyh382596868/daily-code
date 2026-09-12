---
date: 2026-09-02
topic: diffusion
source: trending
repo: Roboparty/UFO
file: humanoidverse/agents/base_model.py
permalink: https://github.com/Roboparty/UFO/blob/0b89378de02f77712072d983bc43ec2a46455dc0/humanoidverse/agents/base_model.py#L1-L94
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, diffusion, checkpoint, load-save, target-network]
---

# UFO BaseModel：checkpoint 保存、加载和 target network 处理 / UFO BaseModel: Checkpoint Save, Load, and Target-Network Handling

> **一句话 / In one line**: UFO 把模型保存、配置落盘和 target network 的懒构建拆成了几个小函数，训练和恢复都很清楚。 / UFO splits model saving, config persistence, and lazy target-network setup into a few small functions, so training and restoration stay explicit.

## 为什么重要 / Why this matters

中文：机器人训练框架最容易乱的地方之一，就是“这个 checkpoint 到底保存了什么、加载时还要补哪些构件”。UFO 用 `save_model()` / `load_model()` 把这件事写死：模型权重进 `safetensors`，构建参数进 JSON 或 pickle，加载时先重建 config，再按 `build_kwargs` 复原对象，最后再决定要不要准备 target network。

English: One of the easiest places for a robotics framework to get messy is checkpoint semantics: what exactly is saved, and what still has to be rebuilt on load? UFO makes that explicit with `save_model()` / `load_model()`: weights go into `safetensors`, build parameters go into JSON or pickle, the config is reconstructed first on load, and target networks are prepared only when needed.

## 代码 / The code

`Roboparty/UFO` — [`humanoidverse/agents/base_model.py`](https://github.com/Roboparty/UFO/blob/0b89378de02f77712072d983bc43ec2a46455dc0/humanoidverse/agents/base_model.py#L1-L94)

```python
import json
import pickle
import typing as tp
from pathlib import Path

import safetensors.torch
import torch
from torch import nn

from .base import BaseConfig
from .envs.utils.gym_spaces import json_to_space, space_to_json


def save_model(path: str, model: "BaseModel", build_kwargs: tp.Optional[tp.Dict[str, tp.Any]] = None) -> None:
    output_folder = Path(path)
    output_folder.mkdir(exist_ok=True)
    safetensors.torch.save_model(model, output_folder / "model.safetensors")

    json_dump = model.cfg.model_dump()

    if build_kwargs is not None:
        if "obs_space" in build_kwargs:
            build_kwargs["obs_space"] = space_to_json(build_kwargs["obs_space"])
        with (output_folder / "init_kwargs.json").open("w+") as f:
            json.dump(build_kwargs, f, indent=4)

    with (output_folder / "config.json").open("w+") as f:
        f.write(json.dumps(json_dump, indent=4))


def load_model(
    path: str, device: str | None, strict: bool, config_class: "BaseModelConfig", build_kwargs: tp.Optional[tp.Dict[str, tp.Any]] = None
) -> "BaseModel":
    model_dir = Path(path)
    with (model_dir / "config.json").open() as f:
        loaded_config = json.load(f)
    if device is not None:
        loaded_config["device"] = device

    if (model_dir / "init_kwargs.pkl").exists():
        with (model_dir / "init_kwargs.pkl").open("rb") as f:
            build_kwargs = pickle.load(f)
    elif (model_dir / "init_kwargs.json").exists():
        with (model_dir / "init_kwargs.json").open("r") as f:
            build_kwargs = json.load(f)
            if "obs_space" in build_kwargs:
                build_kwargs["obs_space"] = json_to_space(build_kwargs["obs_space"])

    if build_kwargs is None:
        raise ValueError(
            "No build_kwargs provided, and init_kwargs.pkl not found. Please provide build_kwargs that are passed to config_class.build functionm."
        )

    loaded_config = config_class(**loaded_config)
    loaded_model = loaded_config.build(**build_kwargs)

    state_dict = safetensors.torch.load_file(model_dir / "model.safetensors", device=device)
    if strict and any(["target" in key for key in state_dict.keys()]):
        loaded_model._prepare_for_train()
    strict = False
    loaded_model.load_state_dict(state_dict, strict=strict)
    return loaded_model


class BaseModelConfig(BaseConfig):
    device: tp.Literal["cpu", "cuda"] = "cuda"


class BaseModel(nn.Module):
    config_class = BaseModelConfig

    def __init__(self, obs_space, action_dim, config: BaseModelConfig):
        super().__init__()
        self.obs_space = obs_space
        self.action_dim = action_dim
        self.cfg = config

    def to(self, *args, **kwargs):
        device, _, _, _ = torch._C._nn._parse_to(*args, **kwargs)
        if device is not None:
            self.device = device.type  # type: ignore
        return super().to(*args, **kwargs)

    @classmethod
    def load(cls, path: str, device: str | None = None, strict: bool = True):
        return load_model(path, device, strict=strict, config_class=cls.config_class)

    def save(self, output_folder: str) -> None:
        return save_model(output_folder, self, build_kwargs={"obs_space": self.obs_space, "action_dim": self.action_dim})
```

## 逐行讲解 / What's happening

1. **第 1-27 行 / Lines 1-27**:
   - 中文: `save_model()` 把权重、配置和构建参数分开保存，避免只留一个“神秘 checkpoint 文件”。
   - English: `save_model()` separates weights, config, and construction kwargs so you do not end up with one opaque checkpoint blob.
2. **第 29-63 行 / Lines 29-63**:
   - 中文: `load_model()` 先读 config，再复原 `build_kwargs`，最后重新 build 模型对象。
   - English: `load_model()` reads the config first, reconstructs `build_kwargs`, then rebuilds the model object.
3. **第 65-94 行 / Lines 65-94**:
   - 中文: `BaseModel.to()` 顺手记住当前 device；`load()` / `save()` 则把类方法和磁盘格式绑定起来。
   - English: `BaseModel.to()` remembers the active device; `load()` / `save()` bind class helpers directly to the on-disk format.

## 类比 / The analogy

中文：像给机器人寄一只装箱的工具箱。箱子里不仅有零件，还有装配说明书；打开时先照说明书重装，再决定要不要把备用零件也装回去。

English: It is like mailing a boxed tool kit for a robot. The box contains the parts plus an assembly manual; when you open it, you rebuild from the manual first and then decide whether to reinstall the spare parts.

## 自己跑一遍 / Try it yourself

```python
saved = {"config": {"device": "cuda"}, "init": {"obs_space": "dict", "action_dim": 4}, "state": {"w": 1}}

def load(saved, device="cpu"):
    cfg = dict(saved["config"])
    cfg["device"] = device
    return {"cfg": cfg, "built_from": saved["init"], "targets_ready": "target" in saved["state"]}

print(load(saved))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
{'cfg': {'device': 'cpu'}, 'built_from': {'obs_space': 'dict', 'action_dim': 4}, 'targets_ready': False}
```

中文：最值得注意的是，`load_model()` 不是单纯读权重，它还负责把“如何构建模型”一起恢复。

English: The important bit is that `load_model()` is not just reading weights; it also restores the recipe for how the model is built.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **PyTorch `state_dict` + config objects** / **PyTorch `state_dict` + config objects**: 中文: 也是把“参数”和“如何实例化模块”分开存。 / English: They also separate “parameters” from “how to instantiate the module.”
- **Hugging Face `save_pretrained()` / `from_pretrained()`** / **Hugging Face `save_pretrained()` / `from_pretrained()`**: 中文: config 驱动的加载习惯几乎同构。 / English: Config-driven loading follows almost the same shape.

## 注意事项 / Caveats / when it breaks

- **`build_kwargs` 不是可选装饰品 / `build_kwargs` is not optional decoration**: 中文: 没有它就没法重建模型。 / English: Without it, the model cannot be rebuilt.
- **`strict` 会影响 target network 的恢复路径 / `strict` affects target-network restoration**: 中文: 这里专门检查了是否有 `target` 键。 / English: The code explicitly checks whether any `target` keys are present.

## 延伸阅读 / Further reading

- UFO source: https://github.com/Roboparty/UFO/blob/0b89378de02f77712072d983bc43ec2a46455dc0/humanoidverse/agents/base_model.py
