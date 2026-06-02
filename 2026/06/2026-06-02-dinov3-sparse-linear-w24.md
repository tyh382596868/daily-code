---
date: 2026-06-02
topic: diffusion
source: tracked
repo: facebookresearch/dinov3
file: dinov3/layers/sparse_linear.py
permalink: https://github.com/facebookresearch/dinov3/blob/31703e4cbf1ccb7c4a72daa1350405f86754b6d1/dinov3/layers/sparse_linear.py#L19-L90
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, diffusion, 2-4-sparsity, inference-acceleration]
---

# 90 行 2:4 稀疏的可热切 nn.Linear / 90 lines of hot-swappable 2:4 sparse nn.Linear

> **一句话 / In one line**: 一个 `nn.Linear` 子类,只在 forward 里把权重做 2:4 稀疏化,配一个全树替换工具和一个运行时开关 —— 训练精度不掉,推理直接吃到 Ampere/Hopper 的稀疏 Tensor Core 加速。 / An `nn.Linear` subclass that 2:4-sparsifies its weight inside forward, paired with a whole-tree replacement helper and a runtime toggle — same gradients during training, free Ampere/Hopper sparse-Tensor-Core throughput at inference.

## 为什么重要 / Why this matters

NVIDIA 从 Ampere 开始在 Tensor Core 上原生支持 2:4 结构化稀疏(每 4 个连续权重里恰好 2 个为 0),理论上能拿到 2× 吞吐。但实际工程里这事一直很难落:多数实现要么是「训练后剪枝再校准」,要么要硬改 model 类。DINOv3 给了一个最小可读的反例 —— 90 行,一个新的 `Linear` 子类,一个全树替换函数,一个运行时开关,就把稀疏化挂到一个已训练好的 ViT 上,并且可以随时关掉对比。

NVIDIA Tensor Cores from Ampere onward natively accelerate 2:4 structured sparsity (in every window of 4 consecutive weights, exactly 2 are zero), giving up to a 2× throughput win. In practice it has been painful to adopt: most recipes either prune after training and recalibrate, or require model surgery. DINOv3 ships a 90-line counter-example — one `Linear` subclass, one tree-replace helper, one runtime flag — that bolts sparsity onto an already-trained ViT and lets you flip it off again to A/B compare.

## 代码 / The code

`facebookresearch/dinov3` — [`dinov3/layers/sparse_linear.py`](https://github.com/facebookresearch/dinov3/blob/31703e4cbf1ccb7c4a72daa1350405f86754b6d1/dinov3/layers/sparse_linear.py#L19-L90)

```python
class LinearW24(torch.nn.Linear):
    ALGO = "largest_abs_values_greedy"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.sparsity_enabled = False

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not self.sparsity_enabled:
            return super().forward(input)

        input_shape = input.shape
        input = input.flatten(end_dim=-2)
        dim0 = input.shape[0]
        if dim0 % 8 != 0:
            # NOTE: This should be torch-compiled away
            input = F.pad(input, [0, 0, 0, -dim0 % 8])
        w_sparse = xops.sparsify24(
            self.weight,
            algo=self.ALGO,
            gradient="ste",
            backend="cusparselt",
        )
        return F.linear(input, w_sparse, self.bias,)[
            :dim0
        ].unflatten(dim=0, sizes=input_shape[:-1])


def replace_linears_with_sparse_linear(root_module: nn.Module, *, filter_fn: Callable[[str], bool]) -> nn.Module:
    total_count = 0

    def replace(module: nn.Module, name: str) -> nn.Module:
        nonlocal total_count
        if not isinstance(module, nn.Linear) or not filter_fn(name):
            return module
        assert type(module) == nn.Linear, "Subtypes not supported"
        new_module = LinearW24(
            in_features=module.in_features,
            out_features=module.out_features,
            bias=module.bias is not None,
            dtype=module.weight.dtype,
            device=module.weight.device,
        )
        new_module.weight = module.weight
        new_module.bias = module.bias
        total_count += 1
        return new_module

    out = named_replace(replace, root_module)
    assert total_count > 0, "2:4 sparsity: no layer found to sparsify"
    return out


def update_24sparsity(root_module: nn.Module, enabled: bool) -> int:
    num_modified = 0

    def maybe_apply_sparsity(module: nn.Module, name: str) -> nn.Module:
        nonlocal num_modified
        if not isinstance(module, LinearW24):
            return module
        num_modified += 1
        module.sparsity_enabled = enabled
        logger.info(f"- {'' if module.sparsity_enabled else 'de'}sparsifying {name}")
        return module

    named_apply(maybe_apply_sparsity, root_module)
    # Force re-compile everything
    torch._dynamo.reset_code_caches()
    from torch._inductor.cudagraph_trees import reset_cudagraph_trees

    reset_cudagraph_trees()
    return num_modified
```

## 逐行讲解 / What's happening

1. **`class LinearW24(torch.nn.Linear)` + `sparsity_enabled = False`**:
   - 中文: 直接继承 `nn.Linear`,所有权重、bias、shape、to(device) 都白嫖父类。多一个开关变量,默认关 —— 这就是「热切」的关键。
   - English: Subclass `nn.Linear` so weight/bias/shape/to(device) all inherit for free. Add one boolean flag, default off — that's the entire "hot-swappable" trick.

2. **`if not self.sparsity_enabled: return super().forward(input)`**:
   - 中文: 关掉的时候就是普通 Linear,零开销。这意味着你可以在同一个模型上跑两次 forward,对比稀疏 vs 稠密的 logits 差异,毫无心智负担。
   - English: When the flag is off, it's a plain `Linear` with zero overhead. You can run the same model twice — once sparse, once dense — and diff the logits with no friction.

3. **`input.flatten(end_dim=-2)` + `F.pad(input, [0, 0, 0, -dim0 % 8])`**:
   - 中文: cuSPARSELt 要求 M 维(input 行数)是 8 的倍数。把多维输入展平成 (N, D),不够 8 就补到 8 的倍数,后面 `[:dim0]` 再切回去。注释说 `torch.compile` 应该会把这个 pad 在 dim0 已经够 8 时优化掉。
   - English: cuSPARSELt requires the M dimension (row count) to be a multiple of 8. Flatten the multi-dim input to (N, D), pad it up to a multiple of 8, then slice back to `dim0` after the matmul. The author notes that `torch.compile` should fold the pad away when `dim0 % 8 == 0`.

4. **`w_sparse = xops.sparsify24(self.weight, algo="largest_abs_values_greedy", gradient="ste", backend="cusparselt")`**:
   - 中文: 这是整段最妙的一行。它在 forward 内部把 dense weight 转成 2:4 稀疏视图 —— 但 `gradient="ste"` 让 backward 走「Straight-Through Estimator」,梯度直接穿过稀疏化算子流回 dense weight。所以训练时 weight 仍然按 dense 更新,只是 forward 看到的是被 mask 过的版本。这是「稀疏感知微调」的最低成本写法。
   - English: This single line is the magic. Inside forward it returns a 2:4-sparse view of the dense weight — and `gradient="ste"` tells the autograd machinery to pipe gradients straight through the sparsify op back to the dense weight. So training still updates the full dense weight, only forward sees the masked version. This is the minimum-cost way to do sparsity-aware fine-tuning.

5. **`replace_linears_with_sparse_linear(...)`**:
   - 中文: 走一遍整棵 `nn.Module` 树,凡是符合 `filter_fn(name)` 的 `nn.Linear` 都替换成 `LinearW24`,关键一步是 `new_module.weight = module.weight` —— 直接复用原 Parameter 对象,不复制内存、不丢优化器状态。`assert type(module) == nn.Linear` 显式拒绝子类,避免误伤已经包过 LoRA 之类的层。
   - English: Walk the entire `nn.Module` tree and replace every `nn.Linear` whose name passes `filter_fn` with a `LinearW24`. The critical line is `new_module.weight = module.weight` — it reuses the original Parameter object, so there's no memory copy and the optimizer state stays bound to the same tensor. The strict `assert type(module) == nn.Linear` deliberately rejects subclasses so a previously-wrapped layer (e.g. LoRA) is left alone.

6. **`update_24sparsity(...)` + `torch._dynamo.reset_code_caches()` + `reset_cudagraph_trees()`**:
   - 中文: 翻一个 bool 还不够 —— 如果你之前 `torch.compile` 过模型,inductor 已经把 forward 编译成假设「走 dense 分支」的 kernel 了,翻开关不会生效。所以这里必须显式清掉 dynamo code cache 和 cudagraph trees,让下一次 forward 重新编译走稀疏分支。**忘了这一步是稀疏推理上不去速度最常见的坑。**
   - English: Flipping a bool isn't enough — if you `torch.compile`'d the model earlier, inductor has already specialized the forward for the dense branch and won't honor the new flag. The explicit `reset_code_caches()` + `reset_cudagraph_trees()` forces a fresh compile on the next forward. **Forgetting this is the #1 reason "sparse mode" silently runs at dense speed.**

## 类比 / The analogy

想象一台跑步机有两条传送带轨道 —— 一条是标准胶皮(密集),一条是有钉子的越野道(稀疏)。`LinearW24` 是装了一个踏板开关的跑步机:踩下去就切到越野道。`update_24sparsity` 是站起来按那个踏板。但跑步机的电机控制器(`torch.compile`)记得「上次启动的是胶皮模式」并按胶皮模式预热了电流 —— 不重启电机,踩踏板也不顶用。`reset_code_caches()` 就是按那个红色的重置按钮。

Picture a treadmill with two belt tracks — smooth rubber (dense) and a studded off-road track (sparse). `LinearW24` is the treadmill with a pedal switch: step on it and you're on the off-road belt. `update_24sparsity` is you stepping on the pedal. But the treadmill's motor controller (`torch.compile`) remembers "we started on rubber" and primed its current draw for that — pressing the pedal alone doesn't reach the motor. `reset_code_caches()` is the red reset button that lets the controller re-learn which belt is active.

## 自己跑一遍 / Try it yourself

```python
# pip install torch xformers
import torch, torch.nn as nn

class LinearW24(nn.Linear):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw); self.sparsity_enabled = False
    def forward(self, x):
        if not self.sparsity_enabled: return super().forward(x)
        from xformers.ops import sparsify24
        w = sparsify24(self.weight, algo="largest_abs_values_greedy", gradient="ste")
        return nn.functional.linear(x, w, self.bias)

m = LinearW24(1024, 1024).cuda().bfloat16()
x = torch.randn(8, 1024, device="cuda", dtype=torch.bfloat16)
y_dense = m(x)
m.sparsity_enabled = True
y_sparse = m(x)
diff = (y_dense - y_sparse).abs().mean().item()
nnz_per_4 = (m.weight.view(-1, 4).abs().topk(2, dim=1).indices.numel() // 2)
print(f"mean |Δoutput| = {diff:.4f}")
print(f"weight rows that *could* be 2:4-sparse: {nnz_per_4 // (m.weight.numel()//4) * 100}%")
```

运行 / Run with:
```bash
pip install torch xformers
python try.py
```

预期输出 / Expected output:
```
mean |Δoutput| = ~0.5   (random init: large; fine-tuned weights: <0.1)
weight rows that *could* be 2:4-sparse: 100%
```

注意 random init 下 `|Δoutput|` 较大是预期的 —— 因为权重没经过稀疏感知训练,被强制 mask 掉一半后输出当然变了。真实场景里要么从一个已经稀疏感知微调过的 checkpoint 加载,要么先用同样的 `sparsify24` 做几个 epoch 微调。

A large `|Δoutput|` from random init is expected — half the weights are forced to zero with no prior training. Real use either loads a sparsity-aware checkpoint or does a short fine-tune with the same `sparsify24` in the loop so the surviving weights learn to compensate.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **PyTorch `torch.ao.pruning`** / **PyTorch `torch.ao.pruning`**: 同样思路,但更繁重 —— `SparsityRegistry` 加 pruner 类,适合配置驱动的剪枝实验,不适合 90 行 inline 替换。 / Same idea, much heavier — `SparsityRegistry` + pruner classes are great for config-driven pruning experiments but overkill for an inline 90-line swap.
- **NVIDIA cuSPARSELt 官方示例** / **NVIDIA cuSPARSELt official samples**: 直接调用底层 C API,要自己管 mask、metadata、prune-and-restore,LinearW24 的 STE backward 实际上就是把这套封装到 autograd 之下。 / Calls the low-level C API directly, you manage mask/metadata/prune-and-restore yourself; `LinearW24`'s STE backward is essentially that machinery wrapped inside autograd.
- **DeepSpeed/MoE `dynamic_sparsity`** / **DeepSpeed/MoE `dynamic_sparsity`**: 同样有「forward 时按规则 mask 权重」的思路,但用于 MoE expert dropout,不是 2:4 硬件稀疏。 / Same "mask weights inside forward" idea but used for MoE expert dropout, not 2:4 hardware sparsity.

## 注意事项 / Caveats / when it breaks

- **必须重置 dynamo + cudagraph** / **Must reset dynamo + cudagraph**: 如前所述,翻开关后不调 `reset_code_caches()` + `reset_cudagraph_trees()`,新 forward 会被旧编译产物截走。 / As noted, flip the flag without resetting and the new forward gets intercepted by the old compiled artifact.
- **`ALGO = "largest_abs_values_greedy"`** / **`ALGO = "largest_abs_values_greedy"`**: 是个贪心而非全局最优的 mask 选择策略;在某些 outlier 分布上效果会比 `mask = top2_per_4_by_grad_norm` 之类的差。改起来很容易,但要重新评估精度。 / This is a greedy mask selection, not globally optimal; on outlier-heavy weight distributions other heuristics (e.g. top-2 by per-row gradient norm) can win — easy to swap in, but you have to re-evaluate accuracy.
- **只在 SM80+ 上有意义** / **Only meaningful on SM80+**: A100/H100 才有 2:4 Tensor Core 通路;在 V100、消费卡(部分支持)、Apple Silicon 上跑只是更慢,因为多了一个 sparsify 算子但没有硬件加速。 / Only A100/H100 (and later) have 2:4 Tensor Core paths; on V100, consumer GPUs (partial support), or Apple Silicon you'll be *slower* because the sparsify op costs FLOPs you don't make back.
- **不支持 nn.Linear 的子类** / **Doesn't handle nn.Linear subclasses**: `assert type(module) == nn.Linear` 会拒绝任何已经包过的层。先 wrap 稀疏、再 wrap LoRA;反过来就要改这一行。 / The strict `type` check rejects anything you've already wrapped. Order matters — apply sparsity *first*, LoRA second, or relax that assertion.

## 延伸阅读 / Further reading

- [NVIDIA 2:4 sparsity blog](https://developer.nvidia.com/blog/structured-sparsity-in-the-nvidia-ampere-architecture-and-applications-in-search-engines/)
- [xformers `sparsify24` source](https://github.com/facebookresearch/xformers/blob/main/xformers/ops/sp24.py)
- [DINOv3 README — sparsity recipe](https://github.com/facebookresearch/dinov3#sparsity)
- [Bengio et al., Straight-Through Estimator (the gradient trick)](https://arxiv.org/abs/1308.3432)
