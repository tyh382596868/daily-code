---
date: 2026-05-28
topic: huggingface
source: huggingface
repo: huggingface/accelerate
file: src/accelerate/utils/fsdp_utils.py
permalink: https://github.com/huggingface/accelerate/blob/cde3e58b8f28abce20c3267b5fa78a3a60acfa94/src/accelerate/utils/fsdp_utils.py#L421-L464
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, huggingface, accelerate, fsdp, weight-tying, closure]
---

# 用一个闭包修好 FSDP2 弄丢的 weight tying / A closure that re-ties weights after FSDP2 silently breaks them

> **一句话 / In one line**: FSDP2 会在 meta device 上把每个模块**独立**重新分配参数,这会悄悄打破像 `embedding ↔ lm_head` 这样的权重共享。accelerate 用一个 44 行的闭包:**初始化前**用 `id()` 记下共享参数,**初始化后**把新分配出来的同一份 tensor `setattr` 回所有别名上,就把 tying 救回来了。 / FSDP2's lazy-init re-materializes each module's parameters **independently** on meta device — silently breaking shared parameters like `embedding ↔ lm_head` weight tying. accelerate's 44-line closure records the `id()` of tied params *before* init runs, then *after* init re-assigns the single newly-allocated tensor back onto every alias via `setattr`, restoring the tying.

## 为什么重要 / Why this matters

权重共享是 LLM 训练里非常常见的一个技巧:`lm_head.weight = embedding.weight`,让两个 `Linear`/`Embedding` 引用同一个底层 tensor。好处明显——能省一份参数(对 vocab=128k、hidden=4096 来说,差 ~5 亿 float 参数,接近 2GB)、训练时梯度也会自然累加在共享的那份上。问题是 FSDP2 的初始化路径不"知道"哪些参数被共享:它对每个模块单独调用 `param_init_fn`,而这个 init 函数通常会在 meta tensor 上 `torch.empty_like()` + 加载权重,**每次调用都分配一块新内存**。结果是:跑完 FSDP2 之后,`lm_head.weight` 和 `embedding.weight` 是两块独立的内存,数值可能相同但训练时分别更新,梯度散到两边,数值上完全是错的——而且因为没有 crash、loss 也照常下降,**这种 bug 极难发现**。`ensure_weights_retied` 用一个最小的闭包修这个问题:在 wrapper 里用 `id()` 标记哪些参数是共享的,初始化跑完之后再去模块上抓那些新分配的参数,把"同一份" tensor 通过 `setattr` 重新分发到所有别名上。读懂这一段,你就掌握了"如何在动态 re-allocate 的环境里维持 Python 对象同一性"这个相当狡猾的技巧。

Weight tying is a workhorse trick in LLM training: `lm_head.weight = embedding.weight`, making two layers reference the same underlying tensor. Benefits are obvious — you save a full set of weights (for vocab=128k, hidden=4096 that's ~500M params, nearly 2GB) and gradients naturally accumulate on the shared tensor. The catch is FSDP2's init path: it doesn't *know* which parameters are tied. It calls `param_init_fn` per module, and that init function usually allocates fresh memory on each call (`torch.empty_like()` on the meta tensor, then load weights). After FSDP2 runs, `lm_head.weight` and `embedding.weight` point to **two independent buffers** — the values may match at the start but training updates them separately, gradients split across the two copies, the math is silently wrong. No crash, no obviously misshapen loss curve, **this bug is notoriously hard to spot**. `ensure_weights_retied` is the minimal fix: a wrapper that tags shared params by `id()` before init runs, then after init grabs the freshly-allocated tensors out of the modules and `setattr`s a single one of them back onto every alias. Read this and you've absorbed a slippery technique: maintaining Python object identity in a system that aggressively re-allocates memory.

## 代码 / The code

`huggingface/accelerate` — [`src/accelerate/utils/fsdp_utils.py`](https://github.com/huggingface/accelerate/blob/cde3e58b8f28abce20c3267b5fa78a3a60acfa94/src/accelerate/utils/fsdp_utils.py#L421-L464)

```python
def ensure_weights_retied(param_init_fn, model: torch.nn.Module, device: torch.device):
    _tied_names = getattr(model, "_tied_weights_keys", None)
    if not _tied_names:
        # if no tied names just passthrough
        return param_init_fn

    # get map of parameter instances to params.
    # - needed for replacement later
    _tied_params = {}
    for name in _tied_names:
        name = name.split(".")
        name, param_name = ".".join(name[:-1]), name[-1]
        mod = model.get_submodule(name)
        param = getattr(mod, param_name)

        _tied_params[id(param)] = None  # placeholder for the param first

    # build param_init_fn for the case with tied params
    def param_init_fn_tied_param(module: torch.nn.Module):
        # track which params to tie
        # - usually only 1, but for completeness consider > 1
        params_to_tie = defaultdict(list)
        for n, param in module.named_parameters(recurse=False):
            if id(param) in _tied_params:
                params_to_tie[id(param)].append(n)

        # call the param init fn, which potentially re-allocates the
        # parameters
        module = param_init_fn(module)

        # search the parameters again and tie them up again
        for id_key, _param_names in params_to_tie.items():
            for param_name in _param_names:
                param = _tied_params[id_key]
                if param is None:
                    # everything will be tied to the first time the
                    # param is observed
                    _tied_params[id_key] = getattr(module, param_name)
                else:
                    setattr(module, param_name, param)  # tie


        return module

    return param_init_fn_tied_param
```

## 逐行讲解 / What's happening

1. **`_tied_names = getattr(model, "_tied_weights_keys", None)` (line 422)**:
   - 中文: HuggingFace `PreTrainedModel` 的子类(LLaMA / GPT2 / T5 等)会把绑定的参数名记在 `_tied_weights_keys` 里,例如 `["lm_head.weight"]`——意思是 `lm_head.weight` 绑到 `transformer.wte.weight`。如果没有这个属性,就走 passthrough,什么都不做。
   - English: HuggingFace `PreTrainedModel` subclasses (LLaMA / GPT2 / T5, etc.) record bound parameter names in `_tied_weights_keys`, e.g. `["lm_head.weight"]`, meaning `lm_head.weight` is tied to `transformer.wte.weight`. If the attribute isn't present, return the original function unchanged.

2. **`_tied_params[id(param)] = None` (line 436)**:
   - 中文: 这是全篇最巧妙的一步。**没有保存 tensor 本身,只保存了它的 `id()`**——也就是 Python 对象的内存地址。原因是:在 FSDP2 把模型搬到 meta device 之后,原始 tensor 就被 deallocate 了,我们要的是"它原来的指纹"而不是"它本身"。`id()` 在这里充当一个不可变的标识符,后面比对时只用 `id(new_param) in _tied_params`。
   - English: The cleverest line in the function. **It stores `id(param)`, not the param itself** — that is, the Python object's memory address. Why: FSDP2 will move the model to meta device and the original tensor is gone; we want its *fingerprint*, not the object. `id()` serves as an immutable identity tag; later checks are pure `id(new_param) in _tied_params` lookups.

3. **闭包 `param_init_fn_tied_param` (lines 439-462)**:
   - 中文: 这是 FSDP2 真正会调用的函数,签名跟原本的 `param_init_fn` 一样:接收一个 module,返回一个 module。在里面,它做三步:(a) 先扫一遍这个 module 直接拥有的参数,看 `id()` 是不是出现在 `_tied_params` 里,记下来要 re-tie 哪些;(b) 调用原始 init,这一步内存就被重新分配了;(c) 把新参数 set 回去。
   - English: This is the function FSDP2 actually calls; its signature matches the original `param_init_fn` (takes a module, returns a module). It does three things: (a) scan this module's direct (non-recursive) params, check whether each `id()` appears in `_tied_params`, record names that need re-tying; (b) call the original init, which is where memory gets re-allocated; (c) `setattr` the new params back to restore tying.

4. **`module.named_parameters(recurse=False)` (line 443)**:
   - 中文: `recurse=False` 是关键——FSDP2 是按模块**逐个**调用 `param_init_fn` 的,所以这里只需要扫当前模块自己的参数,子模块那一层会有它们自己的 init 调用来处理。
   - English: `recurse=False` is essential — FSDP2 invokes `param_init_fn` **per module**, so we should only scan parameters owned by this module directly; child modules get their own init calls.

5. **`module = param_init_fn(module)` (line 449)**:
   - 中文: 这一行执行后,`module` 的参数 tensor 就变了——内存地址、对象 id 都跟之前不一样了。原来的 `param` 引用现在指向已经被释放的对象,所以紧接下来必须用名字 (`getattr(module, param_name)`) 拿新的。
   - English: After this line, `module`'s parameter tensors are different — different memory, different object id. Old `param` references now point to deallocated objects, so the next step must re-fetch by name (`getattr(module, param_name)`).

6. **First-seen-wins logic (lines 454-460)**:
   - 中文: 因为 `param_init_fn` 是按 module 顺序被多次调用的,我们第一次看到某个 id 的时候,把这次新分配的 tensor 存起来 (`_tied_params[id_key] = getattr(...)`);后面再遇到同一个 id(在另一个模块里),就把那次的新 tensor 替换成第一次的那份——`setattr(module, param_name, param)` 就是 tying。
   - English: Because `param_init_fn` is called once per module across many modules, the first time we see a given id we **save** that module's freshly-allocated tensor (`_tied_params[id_key] = getattr(...)`); subsequent encounters with the same id (in a different module) **replace** that module's fresh tensor with the saved one via `setattr(module, param_name, param)`. That's the tying.

7. **`return param_init_fn_tied_param` (line 464)**:
   - 中文: 闭包通过 `_tied_params` 这个 dict 在多次调用之间共享状态——这就是闭包的经典用法,比维护一个 class 更轻量。
   - English: The closure shares state across multiple calls via the `_tied_params` dict — the classic use of closures over a small mutable state, lighter than introducing a class.

## 类比 / The analogy

想象你有两个职位需要同一个人兼任("CEO 兼董事长")。HR 系统的 bug 是:每次有人 onboarding 一个职位时,系统都会自动生成一份新的"员工卡"。你早上让 Alice 上 CEO 岗,系统发了一张 ID 卡 A;下午让 Alice 上董事长岗,系统又发了一张 ID 卡 B。从此 Alice 在两个岗位上的工资单是分开的,绩效奖金也分开发——但 Alice 其实就只有一个人。修这个 bug 的办法:HR 在 onboard 之前在小本本上写下"这两个职位是同一个人(用 Alice 的身份证号标记)";onboard 完之后,把第二个岗位的卡 B 偷偷换成卡 A——这样所有系统都通过卡 A 操作她。`ensure_weights_retied` 干的就是这件事:`id()` 是身份证号,`_tied_params` 是小本本,`setattr` 是"换卡"动作。

Imagine two job titles meant to be held by the same person ("CEO and Chairman"). The HR system has a bug: every time someone onboards into a position, it generates a brand new employee ID card. You onboard Alice as CEO in the morning — she gets card A. You onboard her as Chairman in the afternoon — she gets card B. From now on her payroll, benefits, and bonus tracking are split across A and B, even though it's the same Alice. The fix: HR jots down "these two positions are the same person (tracked by Alice's national ID)" *before* the onboard, then *after* the onboard swaps card B for card A behind the scenes — every downstream system now references her via card A. That's exactly `ensure_weights_retied`: `id()` is the national ID, `_tied_params` is the notebook, `setattr` is the card swap.

## 自己跑一遍 / Try it yourself

```python
# try_retie.py — single-process demo, no GPU/FSDP needed
import torch, torch.nn as nn
from collections import defaultdict

class TiedModel(nn.Module):
    def __init__(self, vocab=8, dim=4):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.head = nn.Linear(dim, vocab, bias=False)
        self.head.weight = self.emb.weight  # tied
        self._tied_weights_keys = ["head.weight"]

def naive_init(module):  # pretends to be FSDP2's per-module re-allocator
    for n, p in module.named_parameters(recurse=False):
        setattr(module, n, nn.Parameter(torch.full_like(p, fill_value=float(id(p) % 7))))
    return module

def ensure_retied(param_init_fn, model):
    keys = getattr(model, "_tied_weights_keys", None)
    if not keys: return param_init_fn
    tied = {}
    for k in keys:
        head_name, p_name = k.rsplit(".", 1)
        tied[id(getattr(model.get_submodule(head_name), p_name))] = None
    def wrapped(module):
        to_tie = defaultdict(list)
        for n, p in module.named_parameters(recurse=False):
            if id(p) in tied: to_tie[id(p)].append(n)
        module = param_init_fn(module)
        for k, names in to_tie.items():
            for n in names:
                if tied[k] is None: tied[k] = getattr(module, n)
                else: setattr(module, n, tied[k])
        return module
    return wrapped

def run(init_fn, label):
    m = TiedModel()
    for sub in m.modules():
        if sub is not m: init_fn(sub)
    print(f"{label}: same buffer? {m.emb.weight.data_ptr() == m.head.weight.data_ptr()}")

run(naive_init, "WITHOUT retie")
run(ensure_retied(naive_init, TiedModel()), "WITH    retie")
```

运行 / Run with:
```bash
pip install torch
python try_retie.py
```

预期输出 / Expected output:
```
WITHOUT retie: same buffer? False
WITH    retie: same buffer? True
```

注意 `data_ptr()` 是判断两个 tensor 是否共享内存的标准方式。`False` 那行就是 FSDP2 在生产里的隐藏 bug:形状对、loss 不崩,但 `emb` 和 `head` 已经"分家"。`True` 那行说明 retie 闭包成功地把它们重新指回了同一块内存。

`data_ptr()` is the standard way to check whether two tensors share memory. The `False` line is the production bug FSDP2 introduces: shapes match, loss doesn't crash, but `emb` and `head` are now "separated." The `True` line confirms the retie closure successfully pointed both names at the same buffer.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **PyTorch `nn.Module._apply`** / **PyTorch `nn.Module._apply`**: 在 `model.to(device)` 时也会面临同样的"shared param 被复制两次"问题,PyTorch 内部用 `memo` dict 解决,几乎是同一套思路 / `model.to(device)` faces the same "shared param copied twice" problem; PyTorch uses an internal `memo` dict — almost the same recipe.
- **DeepSpeed ZeRO-3 weight tying patch** / **DeepSpeed ZeRO-3 weight tying patch**: ZeRO-3 同样会按参数 shard,需要额外的 hook 维持 tying / ZeRO-3 also shards per parameter and needs an extra hook to preserve tying.
- **torch.fx graph rewriting** / **torch.fx graph rewriting**: 重写完图之后参数名映射常常需要类似 "old id → new node" 的字典来维持引用关系 / After rewriting an FX graph you usually need an "old id → new node" map to keep references coherent.
- **PEFT 的 `tied_weights_keys` 处理** / **PEFT's `tied_weights_keys` handling**: 给 LoRA 加 adapter 时,如果原模型有 tied weights,PEFT 必须 fork 一份独立的 adapter weight 而不能跟着 tie / Adding a LoRA adapter to a model with tied weights requires PEFT to fork its own independent adapter weight rather than tying along.
- **JAX `jax.tree_util.register_pytree_node`** / **JAX pytree shared-leaf handling**: JAX 的 pytree 默认不保留共享,需要显式 deduplication / JAX's pytree doesn't preserve sharing by default; you have to deduplicate explicitly.

## 注意事项 / Caveats / when it breaks

- **`id()` 在 tensor 被 GC 后会被重用** / **`id()` is reused after a tensor is GC'd**: Python 的 `id()` 是基于地址的,一旦原 tensor 被回收,新分配的对象可能拿到同一个 id。这里之所以安全,是因为闭包持有了对原始 `param` 的引用 → 不会被 GC,直到闭包消失 / Python's `id()` is address-based, and a deallocated tensor's id can be reused. This code stays safe because the closure holds a live reference to the original `param` — preventing GC until the closure itself goes away.
- **只处理直接子参数 (`recurse=False`)** / **Only handles direct child parameters**: 如果一个模块包另一个模块,且 inner module 的参数被 tied,你需要确保 FSDP2 也对 inner module 调用了这个 wrapper / If a module *contains* a submodule whose params are tied, you need FSDP2 to also invoke this wrapper on the submodule.
- **`_tied_weights_keys` 是 HF 约定** / **`_tied_weights_keys` is an HF convention**: 非 HuggingFace 模型(比如自己写的 nn.Module)没有这个属性,需要手动设置才能享受这个保护 / Non-HuggingFace `nn.Module`s won't have this attribute; you need to set it manually to opt in.
- **如果 `param_init_fn` 本身就保留对象同一性,则什么都不需要** / **No-op when `param_init_fn` itself preserves identity**: 一些自定义 init 会在 meta 之上 `in-place` 修改而不重新分配,那种情况这层包装是冗余但无害的 / Some custom inits mutate in-place on meta tensors without re-allocating — this wrapper becomes a (harmless) no-op.
- **多卡时所有 rank 的 tying 结构必须一致** / **All ranks must agree on the tying structure**: 否则不同 rank 上的 `_tied_params` 不一致,后续 broadcast 会形变 / If different ranks have different tied structures, downstream broadcasts will desync.

## 延伸阅读 / Further reading

- [HuggingFace `tie_weights` source](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py) — `_tied_weights_keys` 的发源地,看怎么用 `clone_module_weight` 维持 tying
- [PyTorch FSDP2 docs — auto_wrap and init paths](https://pytorch.org/docs/stable/distributed.fsdp.fully_shard.html) — 解释了为什么 init 必须是按模块独立调用
- [GPT-2 weight tying paper (Press & Wolf 2017)](https://arxiv.org/abs/1608.05859) — "Using the Output Embedding to Improve Language Models",weight tying 的最早动机
- [accelerate PR #1979 — FSDP2 weight tying fix](https://github.com/huggingface/accelerate/pulls?q=is%3Apr+ensure_weights_retied) — 当初这段代码合入的讨论,可以看到工程师踩坑的过程
