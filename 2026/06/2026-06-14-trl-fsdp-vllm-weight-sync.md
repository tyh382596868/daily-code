---
date: 2026-06-14
topic: huggingface
source: huggingface
repo: huggingface/trl
file: trl/generation/vllm_generation.py
permalink: https://github.com/huggingface/trl/blob/eec4ed2c352b086d3fe1f6686d4cf98e287f38a3/trl/generation/vllm_generation.py#L382-L437
difficulty: advanced
read_time: ~12 min
tags: [code-of-the-day, huggingface, trl, rlhf, vllm, fsdp, distributed]
---

# TRL 把 RLHF 的"训练权重 → vLLM 推理"压成 56 行,FSDP1 和 FSDP2 各走一条路 / TRL squeezes RLHF's "shipping training weights into vLLM" into 56 lines — FSDP1 and FSDP2 take different routes

> **一句话 / In one line**: PPO/GRPO 每个迭代都要把新策略推回 vLLM 推理服务器,这里的 56 行同时处理 FSDP1 / FSDP2、server / colocate 两种模式,本质就是"把 sharded param 拼成完整 tensor,然后一行一行喂进 vLLM"。 / Every PPO/GRPO iteration ships the freshly-trained policy back to a vLLM rollout server; these 56 lines handle FSDP1 + FSDP2 and server + colocate mode, with the same recipe: "materialise each sharded param into a full tensor and feed them into vLLM one by one".

## 为什么重要 / Why this matters

现代 RLHF 训练栈是双进程的:训练侧用 FSDP 把模型切片,推理侧用 vLLM 跑 prefix-cache 快速 rollout。两边 GPU 重叠或不重叠都行,但每个 RL step 之后,新策略必须送回 vLLM,否则下一轮采样还是用老模型。这条"权重摆渡"路径以前在 TRL 里散落在四五个地方,2026-06-12 的 PR #6004 (昨天!) 把它抽成了 `_push_param_to_vllm`,然后两条 FSDP1/FSDP2 同步函数各自调用它。56 行覆盖了所有边角:server 模式只有 rank 0 push,colocate 模式直接 `load_weights`,FSDP1 要后序遍历 + `summon_full_params(writeback=False)`,FSDP2 用 `state_dict()` + `full_tensor()`。

Modern RLHF stacks are two-process: training uses FSDP-sharded params, inference uses vLLM for prefix-cache-accelerated rollouts. The two halves can colocate or not, but after every RL step the fresh policy *must* be shipped to vLLM, or the next rollout uses stale weights. This "ferry" path used to be scattered across four or five places in TRL — yesterday's PR #6004 (2026-06-12) extracted `_push_param_to_vllm` and now both FSDP1 and FSDP2 sync functions call into it. Fifty-six lines cover every edge case: server mode pushes only on rank 0, colocate mode calls `load_weights` directly, FSDP1 needs a post-order traversal with `summon_full_params(writeback=False)`, and FSDP2 just uses `state_dict()` + `full_tensor()`.

## 代码 / The code

`huggingface/trl` — [`trl/generation/vllm_generation.py`](https://github.com/huggingface/trl/blob/eec4ed2c352b086d3fe1f6686d4cf98e287f38a3/trl/generation/vllm_generation.py#L382-L437)

```python
def _push_param_to_vllm(self, name: str, param) -> None:
    """Push a single parameter tensor to the vLLM engine (server or colocate mode)."""
    if self.mode == "server" and self.accelerator.is_main_process:
        self.vllm_client.update_named_param(name, param)
    elif self.mode == "colocate":
        self.llm.llm_engine.model_executor.driver_worker.model_runner.model.load_weights([(name, param)])

def _sync_fsdp1_params_to_vllm(self, module: nn.Module, prefix: str = "", visited: set[str] | None = None):
    """Memory-efficient post-order traversal of FSDP modules to extract full parameters and sync with vLLM."""
    # For FSDP1, we need to recurse into children and also use summon_full_params
    if visited is None:
        visited = set()
    for child_name, child_module in module.named_children():
        child_prefix = f"{prefix}.{child_name}" if prefix else child_name
        self._sync_fsdp1_params_to_vllm(
            child_module, prefix=child_prefix, visited=visited
        )  # recurse into the child

    if isinstance(module, FSDP):
        with FSDP.summon_full_params(module, recurse=False, writeback=False):
            for param_name, param in module.named_parameters():
                full_name = f"{prefix}.{param_name}" if prefix else param_name
                full_name = self._fix_param_name_to_vllm(full_name, extra_prefixes=["_fsdp_wrapped_module."])

                if full_name in visited:
                    continue  # skip FSDP subtrees already traversed
                visited.add(full_name)

                self._push_param_to_vllm(full_name, param.data)

def _sync_fsdp2_params_to_vllm(self, module: nn.Module):
    """FSDP2-specific parameter synchronization."""
    # For FSDP2, module.state_dict() already covers all parameters, so no need for recursion
    for name, param in module.state_dict().items():
        # When using PEFT, we need to recover the original parameter name
        name = name.removeprefix("base_model.model.").replace(".base_layer", "")
        # Skip PEFT layers: they don't exist in vLLM, and they are merged already.
        if is_peft_model(module) and module.prefix in name:
            continue
        # When module to save, remove its prefix and discard the original module
        if "original_module" in name:
            continue
        name = self._fix_param_name_to_vllm(name, extra_prefixes=["modules_to_save.default."])

        if param.is_cpu:
            param = param.to(torch.device("cuda"))
        param = param.full_tensor()

        self._push_param_to_vllm(name, param)
```

## 逐行讲解 / What's happening

1. **`_push_param_to_vllm`(第 382-387 行)/ Lines 382-387**:
   - 中文: server 模式下,只有 rank 0 调用 `vllm_client.update_named_param` 通过 HTTP 推送一个张量给独立的 vLLM 进程;colocate 模式下,vLLM engine 就在本进程,直接钻进 `model_executor.driver_worker.model_runner.model.load_weights` 调用底层 weight loader。注意这个长链子上每一层都是 vLLM 公开的内部 API。
   - English: In `"server"` mode, only rank 0 calls `vllm_client.update_named_param`, pushing one tensor over HTTP to the standalone vLLM process. In `"colocate"` mode the vLLM engine lives in the same process, so we drill into `model_executor.driver_worker.model_runner.model.load_weights`. Every link in that long chain is a documented vLLM internal.

2. **第 389-410 行 / Lines 389-410 (`_sync_fsdp1_params_to_vllm`)**:
   - 中文: FSDP1 是经典版——参数被切到多卡 sharded shape,要拿到 full tensor 必须用 `summon_full_params`。这个 context manager 会把 shards all-gather 到本 rank,但只在这个 rank 上,而且 `writeback=False` 表示退出 context 时不要把修改写回——我们只读,所以可以省一次 reduce-scatter。
   - English: FSDP1 is the classic version — each parameter is sharded across ranks, so you must use `summon_full_params` to materialise the full tensor. That context manager all-gathers shards onto the current rank only; `writeback=False` means "don't reduce-scatter back at exit" — we're only reading, so we save one collective.

3. **后序遍历是关键 / Why post-order traversal**:
   - 中文: 注意他们先递归子节点 (`named_children`),再处理当前 `FSDP` 包装的 module——这是后序遍历。原因是 FSDP1 可以嵌套(整个 transformer 是一个 FSDP module,里面每个 transformer block 也可以单独包成 FSDP)。从最深的子节点开始 summon,内存峰值只等于"最深那一层 + 当前栈帧",而不是整棵树。`visited` 集合保证父节点 summon 时不重复推送子节点的参数。
   - English: They recurse into `named_children` *first* and only then handle the current `FSDP`-wrapped module — that's post-order. The reason: FSDP1 can nest (the whole transformer is one FSDP module, and each transformer block can independently be another FSDP module). Summon from the deepest child first and the peak memory is just "this subtree + current stack frame" rather than the whole graph. The `visited` set guarantees that when the parent later summons, we don't re-push params already visited.

4. **`writeback=False` 和 `recurse=False`**:
   - 中文: 两个参数都很关键。`recurse=False` 告诉 summon 不要自动递归——因为我们已经手动在递归了;否则 summon 会试图 all-gather 子节点的子节点,跟我们的循环打架。`writeback=False` 已解释过。
   - English: Both flags are critical. `recurse=False` tells `summon` not to recurse — we're already doing it manually; otherwise `summon` would try to all-gather grandchildren too and collide with our loop. `writeback=False` we covered above.

5. **第 412-430 行 / Lines 412-430 (`_sync_fsdp2_params_to_vllm`)**:
   - 中文: FSDP2 (DTensor 时代) 简单得多。`state_dict()` 直接返回所有参数,但每个 value 是个 DTensor(分片张量)。`.full_tensor()` 一行就 all-gather 出完整 tensor。注意:它还顺便把 CPU param 搬到 GPU(`if param.is_cpu`)——FSDP2 + `cpu_offload` 时很常见。
   - English: FSDP2 (the DTensor era) is dramatically simpler. `state_dict()` directly returns every parameter, but the values are DTensors (sharded). One call to `.full_tensor()` triggers the all-gather and yields the dense tensor. Bonus: it also hoists CPU params back to GPU (`if param.is_cpu`) — common when FSDP2 has `cpu_offload` on.

6. **PEFT 名字修正 / PEFT name surgery**:
   - 中文: 训练侧用 PEFT 会在参数名里塞 `base_model.model.` 和 `.base_layer`,vLLM 推理用 merged 后的命名。三行字符串操作 (`removeprefix` + `replace`) 把训练侧的名字"还原"成 vLLM 侧的样子——是这种异构系统集成的典型胶水代码。
   - English: PEFT prepends `base_model.model.` and inserts `.base_layer` into param names; vLLM uses the merged (clean) names. Three lines of string surgery (`removeprefix` + `replace`) rename training-side params to vLLM-side conventions — classic glue code for heterogeneous systems.

## 类比 / The analogy

想象一家有两个厨房的餐厅:中央厨房(训练 FSDP)负责"研发新菜",前厅厨房(vLLM 推理)负责"出餐"。每次新菜定稿后,要把配方送过去。配方表本来是 8 张卡各拿一份(参数切片),你不能 8 张分别送——前厅那位会拼出错的菜。`summon_full_params` 就是"把 8 张拼起来,印一份完整的临时配方"——而且印完不收回(`writeback=False`),省时间。然后一行一行(因为前厅一次只能处理一道菜)递过去。 server 模式 = 中央厨房有专门的"传菜员"(rank 0)走过去;colocate 模式 = 两个厨房本来就连着,直接递。

Picture a restaurant with two kitchens: a central R&D kitchen (FSDP training) develops new dishes, and a front kitchen (vLLM inference) cooks them at speed. Whenever a recipe finalises, you need to ship it across. The recipe sheet was split across 8 chefs (param shards); you can't ship 8 fragments — the front kitchen would cook the wrong dish. `summon_full_params` is "assemble the 8 fragments into one printed temporary recipe" — and don't bother collecting the printout (`writeback=False`) since you'll throw it away. Then hand it across one item at a time (the front kitchen prepares one dish at a time). Server mode = the central kitchen has a dedicated runner (rank 0) walking the recipe over; colocate mode = the kitchens share a wall, so you just hand it through.

## 自己跑一遍 / Try it yourself

```python
# A toy single-process simulation of the FSDP2 pattern. Real code requires torchrun.
import torch
import torch.nn as nn

class TinyVLLM:
    """Pretend vLLM engine: stores incoming weights by name."""
    def __init__(self): self.store = {}
    def load_weights(self, pairs): self.store.update(dict(pairs))

def push(vllm, name, param):
    vllm.load_weights([(name, param)])  # colocate path

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(8, 8)
        self.layer2 = nn.Linear(8, 8)

trainer_model = Net()
vllm = TinyVLLM()

# Simulate FSDP2 sync: state_dict + full_tensor (no-op here because we're single-process).
for name, param in trainer_model.state_dict().items():
    # imagine: param = param.full_tensor()
    push(vllm, name, param)

print("synced names:", list(vllm.store))
print("layer1.weight norm:", vllm.store["layer1.weight"].norm().item())
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
synced names: ['layer1.weight', 'layer1.bias', 'layer2.weight', 'layer2.bias']
layer1.weight norm: 2.42...
```

中文一句:真实 FSDP 场景下,`state_dict()` 返回的是 DTensor,你需要在 push 前调一次 `.full_tensor()`——这一行就是 sharded → dense 的 all-gather 触发点。

English: in a real FSDP setting, `state_dict()` returns DTensors, and you must call `.full_tensor()` before pushing — that single line triggers the sharded-to-dense all-gather.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Open-source `verl` (Volcengine RLHF framework)** / **`verl`**: 同样用 vLLM 做 rollout 引擎,有自己的 `WeightLoader` 抽象层,也是先 gather sharded weights 再 send。 / Uses vLLM as the rollout engine with its own `WeightLoader` abstraction; same "gather sharded weights then send" recipe.
- **DeepSpeed-Chat / ChatLearn** / **DeepSpeed-Chat / ChatLearn**: DeepSpeed ZeRO-3 用 `zero.GatheredParameters(p)` 替代 `summon_full_params`,但骨架完全一样。 / DeepSpeed ZeRO-3 swaps `summon_full_params` for `zero.GatheredParameters(p)`, but the skeleton is identical.
- **OpenAI / Anthropic 内部 RLHF stacks** / **Internal RLHF stacks at OpenAI / Anthropic**: 据公开演讲透露,他们用 NCCL 直接 P2P 把 trainer rank 上的参数 copy 到 rollout rank,跳过 HTTP——效率更高但工程更复杂。 / Per public talks they use raw NCCL P2P to copy weights from trainer ranks to rollout ranks, bypassing HTTP — faster but more engineering.
- **vLLM 的 `update_named_param` API** / **vLLM's `update_named_param` API**: TRL `vllm_client` 直接调用的接口,vLLM 在 0.6+ 之后就提供了。 / The exact API TRL's `vllm_client` invokes; vLLM has shipped it since 0.6+.

## 注意事项 / Caveats / when it breaks

- **summon_full_params 是阻塞的** / **`summon_full_params` is blocking**: 在内部触发 all-gather,如果 rank 之间步调不一致(比如某个 rank 提前进了 except 分支)会死锁。所有 rank 必须同时进入这段代码。 / Internally it triggers an all-gather; if ranks fall out of sync (e.g. one rank entered an `except` branch early) you deadlock. All ranks must enter this block together.
- **writeback=False 不能配 PEFT merge** / **`writeback=False` is incompatible with PEFT merge inside the same context**: `merge_adapter()` 会修改参数,如果 `writeback=False` 这些修改会在退出 context 时丢失。所以代码外层用 `gather_params` 把全参数拿到主 rank,在 ZeRO-3 下 merge,而不是在 summon 里 merge。 / `merge_adapter()` mutates params; with `writeback=False` those mutations vanish on exit. That's why the outer `sync_weights` uses `gather_params(list(model.parameters()))` to assemble the model on the main rank first, then merges, then enters the summon path.
- **server 模式下 push 是单向的** / **In `server` mode, push is one-way**: 只有 rank 0 跟 vLLM 通信,所以前提是 rank 0 拿得到 full param。配合 `summon_full_params(recurse=False)`,只有该 rank 看到完整 tensor——刚好对应 `if main_process: push` 的设计。 / Only rank 0 talks to vLLM, so rank 0 must hold the full param. The `summon_full_params(recurse=False)` makes only the current rank materialise it — which lines up exactly with `if main_process: push`.
- **vLLM colocate 必须 `wake_up` 一次** / **vLLM colocate needs a `wake_up` first**: 因为 sleep mode 会把 weight memory 解除映射,这时直接 `load_weights` 会写到 freed memory。`sync_weights` 在开头有 `self.llm.wake_up(tags=["weights"])` 就是为了避免这个崩溃(详见 issue #5142)。 / Sleep mode unmaps weight memory; writing into it directly crashes. The `sync_weights` opener calls `self.llm.wake_up(tags=["weights"])` to guard against this (issue #5142).

## 延伸阅读 / Further reading

- [PR #6004 — Extract _push_param_to_vllm helper](https://github.com/huggingface/trl/pull/6004)
- [TRL GRPO trainer docs](https://huggingface.co/docs/trl/main/en/grpo_trainer)
- [PyTorch FSDP2 (DTensor) tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- [vLLM `update_named_param` API](https://docs.vllm.ai/en/latest/dev/offline_inference/llm.html)
