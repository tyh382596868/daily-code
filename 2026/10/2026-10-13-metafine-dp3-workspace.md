---
date: 2026-10-13
topic: diffusion
source: trending
repo: Hiangx-robotics/MetaFine
file: core/policies/diffusion_policy_3d/train.py
permalink: https://github.com/Hiangx-robotics/MetaFine/blob/9d23841684295abc3a4163c6a5918e2590555ffa/core/policies/diffusion_policy_3d/train.py#L42-L146
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, diffusion, diffusion-policy, workspace]
---

# MetaFine DP3 workspace：训练状态、normalizer 和 scheduler 一起落座 / MetaFine DP3 Workspace: Seat Training State, Normalizer, and Scheduler Together

> **一句话 / In one line**: MetaFine 的 DP3 workspace 在训练前统一注册环境、设 seed、绑定 normalizer，并按梯度累积计算 scheduler 步数。 / MetaFine's DP3 workspace registers environments, seeds RNGs, binds the normalizer, and computes scheduler steps with gradient accumulation before training.

## 为什么重要 / Why this matters

Diffusion policy 训练常被环境注册、normalizer、EMA 和 scheduler 步数这些边界条件拖乱；workspace 把它们集中到入口。

Diffusion policy training is often derailed by environment registration, normalizers, EMA state, and scheduler step counts; the workspace centralizes those boundaries.

## 代码 / The code

`Hiangx-robotics/MetaFine` - [`core/policies/diffusion_policy_3d/train.py`](https://github.com/Hiangx-robotics/MetaFine/blob/9d23841684295abc3a4163c6a5918e2590555ffa/core/policies/diffusion_policy_3d/train.py#L42-L146)

```python
def _register_custom_envs():
    """Register custom environments from git/core/env.py."""
    try:
        git_root = str(pathlib.Path(__file__).resolve().parents[3])
        if git_root not in sys.path:
            sys.path.append(git_root)
        import core.env  # noqa: F401
    except Exception as e:
        print(f"[Warn] Could not import core.env for env registration: {e}")


# Important: eval.py imports TrainDP3Workspace from this module.
# Register envs at import time so gym.make() can find custom env IDs.
_register_custom_envs()


class TrainDP3Workspace:
    include_keys = ['global_step', 'epoch']
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf, output_dir=None):
        self.cfg = cfg
        self._output_dir = output_dir
        self._saving_thread = None

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DP3 = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DP3 = None
        if cfg.training.use_ema:
            try:
                self.ema_model = copy.deepcopy(self.model)
            except:  # minkowski engine could not be copied. recreate it
                self.ema_model = hydra.utils.instantiate(cfg.policy)

        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        # configure training state
        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        if cfg.training.debug:
            cfg.training.num_epochs = 100
            cfg.training.max_train_steps = 10
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 20
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1
            RUN_ROLLOUT = True
            RUN_CKPT = False
            verbose = True
        else:
            RUN_ROLLOUT = True
            RUN_CKPT = True
            verbose = False

        RUN_VALIDATION = False  # reduce time cost

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: BaseDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)

        assert isinstance(dataset, BaseDataset), print(f"dataset must be BaseDataset, got {type(dataset)}")
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                                       len(train_dataloader) * cfg.training.num_epochs) \
                               // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step - 1
        )
```

## 逐行讲解 / What's happening

1. **第 42-55 行 / Lines 42-55 (env registration)**:
   - 中文: import 时注册自定义 env，eval.py 复用 workspace 时不会找不到 Gym id。
   - English: It registers custom envs at import time so eval.py can reuse the workspace without missing Gym IDs.
2. **第 62-89 行 / Lines 62-89 (state construction)**:
   - 中文: 保存 config、设置随机种子、实例化 DP3、创建 EMA/optimizer/计数器。
   - English: It stores config, seeds RNGs, instantiates DP3, and creates EMA, optimizer, and counters.
3. **第 119-134 行 / Lines 119-134 (normalizer)**:
   - 中文: normalizer 来自数据集，并同时写入主模型和 EMA 模型。
   - English: The normalizer comes from the dataset and is installed into both main and EMA models.
4. **第 135-146 行 / Lines 135-146 (schedule length)**:
   - 中文: 总训练步数按 batch、epoch 和 gradient accumulation 计算。
   - English: Total training steps are computed from batches, epochs, and gradient accumulation.

## 类比 / The analogy

这像开烘焙流水线：先校准秤、设温度、定传送带速度，再开始烤。

It is like starting a bakery line: calibrate the scale, set temperature, choose conveyor speed, then bake.

## 自己跑一遍 / Try it yourself

```python
def scheduler_steps(num_batches, num_epochs, grad_accum):
    return (num_batches * num_epochs) // grad_accum
normalizer = {"action": "mean/std from dataset"}
models = {"policy": {}, "ema": {}}
for model in models.values():
    model["normalizer"] = normalizer
print(scheduler_steps(120, 10, 4))
print(models["policy"]["normalizer"] is models["ema"]["normalizer"])
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
300
True
```

中文: scheduler 跟 batch 更新次数对齐，normalizer 同时喂给主模型和 EMA。

English: The scheduler matches batch update count, and the same normalizer feeds main and EMA models.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Diffusion Policy workspaces** / **Diffusion Policy workspaces**: Diffusion Policy workspaces。 / Diffusion Policy workspaces.
- **LeRobot training pipelines** / **LeRobot training pipelines**: LeRobot training pipelines。 / LeRobot training pipelines.

## 注意事项 / Caveats / when it breaks

- **import side effects** / **Missing env dependencies become runtime warnings**: import side effects。 / Missing env dependencies become runtime warnings.
- **EMA copy fallback** / **Re-instantiated EMA models still need correct weight sync**: EMA copy fallback。 / Re-instantiated EMA models still need correct weight sync.

## 延伸阅读 / Further reading
- [MetaFine DP3 workspace](https://github.com/Hiangx-robotics/MetaFine/blob/9d23841684295abc3a4163c6a5918e2590555ffa/core/policies/diffusion_policy_3d/train.py#L42-L146)
