---
date: 2026-05-28
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/distributed/_composable/checkpoint_activation.py
permalink: https://github.com/pytorch/pytorch/blob/fd6d216e3e8bf07c470716dfbf022d82fadd521d/torch/distributed/_composable/checkpoint_activation.py#L16-L135
difficulty: advanced
read_time: ~13 min
tags: [code-of-the-day, pytorch, activation-checkpoint, generator, hooks, composable-api]
---

# 用 forward hook + generator 实现可组合的 activation checkpointing / Composable activation checkpointing with forward hooks and a generator

> **一句话 / In one line**: PyTorch 的新 `torch.distributed._composable.checkpoint` 不再包模块、不再改 forward——它用 `forward_pre_hook` 调 `next(gen)` 进入 checkpoint 上下文,用 `forward_hook` 再调 `next(gen)` 触发 `StopIteration` 退出。整套 activation checkpointing 就是一个一次性 generator 的两次 `next`。 / PyTorch's new `torch.distributed._composable.checkpoint` doesn't wrap the module, doesn't rewrite forward — instead, `forward_pre_hook` enters the checkpoint context via `next(gen)` and `forward_hook` exits it by calling `next(gen)` again and catching the expected `StopIteration`. The whole activation-checkpointing dance is two `next()` calls on a one-shot generator.

## 为什么重要 / Why this matters

经典 activation checkpointing 有两个流派:**函数式** (`torch.utils.checkpoint.checkpoint(fn, *args)`),需要改写 forward 把每个 block 包起来;**Wrapper 式** (`CheckpointWrapper(module)`),不改 forward 但把模块名字空间打乱了——本来是 `model.layers.0` 的现在变成了 `model.layers.0._checkpoint_wrapped_module`,各种 state_dict / FSDP / Pipeline Parallel 集成全得跟着改。新的 composable API 既不改 forward,也不动 FQN(fully-qualified names),就在原模块上注册两个 hook 就行——`checkpoint(model.l1)` 一行调用就让 `l1` 的激活在反向时重新计算。它的实现关键是用一个 Python generator 把"进入 / 退出 checkpoint 上下文"序列化:`yield` 之前的部分对应 pre-hook,`yield` 之后的部分对应 post-hook。generator 天然就是"两阶段上下文管理器"的最简表达——这一招对编写任何"前后包夹模块行为"的 API 都有借鉴价值。

Classic activation checkpointing comes in two flavors. **Functional** (`torch.utils.checkpoint.checkpoint(fn, *args)`) needs you to rewrite forward to wrap each block. **Wrapper-based** (`CheckpointWrapper(module)`) leaves forward alone but pollutes the FQN tree — `model.layers.0` becomes `model.layers.0._checkpoint_wrapped_module`, breaking every downstream integration (state_dict, FSDP, pipeline parallel). The composable API touches neither forward nor FQNs: registering two hooks on the original module is enough — `checkpoint(model.l1)` is the entire user-facing call to make `l1` recompute its activations during backward. The implementation trick is Python generators as serialized "enter/exit" pairs: code before `yield` runs in the pre-hook, code after `yield` runs in the post-hook. A generator is literally the minimal form of a two-stage context manager — a pattern worth absorbing for anything that needs to bracket module behavior.

## 代码 / The code

`pytorch/pytorch` — [`torch/distributed/_composable/checkpoint_activation.py`](https://github.com/pytorch/pytorch/blob/fd6d216e3e8bf07c470716dfbf022d82fadd521d/torch/distributed/_composable/checkpoint_activation.py#L16-L135)

```python
@contextmanager
def _no_hook(module: nn.Module, user_ctx: AbstractContextManager | None = None):
    r"""
    Disable hooks installed by checkpoint to avoid unintentional recursion
    during backward recomputation.
    """

    with user_ctx if user_ctx else nullcontext():
        orig_enable_hook = checkpoint.state(module).enable_hook
        checkpoint.state(module).enable_hook = False
        try:
            yield
        finally:
            checkpoint.state(module).enable_hook = orig_enable_hook


class _CheckpointState(_State):
    enable_hook: bool = False
    _ac_generator: Generator[None, None, None] | None


@contract(_CheckpointState)
def checkpoint(module: nn.Module, **kwargs) -> nn.Module:
    r"""
    This is a composable activation checkpointing API. Unlike functional
    activation checkpointing APIs, this one does not require changing model
    source code. Unlike ``nn.Module`` wrapper activation checkpointing APIs,
    this one does not modify model structure or fully-qualified names either.
    Under the hood, it registers activation checkpointing logic as pre- and
    post-forward hooks. Hence, this API can be easily applied to any model or
    sub-modules in the model.
    """
    torch._C._log_api_usage_once("torch.distributed.checkpoint")

    use_reentrant = kwargs.pop("use_reentrant", False)
    if use_reentrant:
        raise NotImplementedError(
            "use_reentrant=True is not supported in composable checkpoint. "
            "Please use torch.utils.checkpoint.checkpoint instead."
        )
    preserve_rng_state = kwargs.pop("preserve_rng_state", True)
    user_context_fns = kwargs.pop("context_fn", None)
    determinism_check = kwargs.pop("determinism_check", _DEFAULT_DETERMINISM_MODE)
    debug = kwargs.pop("debug", False)
    early_stop = kwargs.pop("early_stop", True)

    if kwargs:
        raise ValueError(
            "Unexpected keyword arguments: " + ",".join(arg for arg in kwargs)
        )

    def forward_pre_hook(
        module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> None:
        if checkpoint.state(module).enable_hook:

            def context_fns():
                if user_context_fns is not None:
                    ctx1, ctx2 = user_context_fns()
                    return ctx1, _no_hook(module, ctx2)
                else:
                    return nullcontext(), _no_hook(module)

            gen = _checkpoint_without_reentrant_generator(
                module,
                preserve_rng_state,
                context_fns,
                determinism_check,
                debug,
                early_stop,
                *args,
                **kwargs,
            )
            checkpoint.state(module)._ac_generator = gen
            next(gen)

    def forward_hook(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> Any:
        if checkpoint.state(module).enable_hook:
            try:
                gen = checkpoint.state(module)._ac_generator
                if gen is None:
                    raise AssertionError
                next(gen)
            except StopIteration:
                pass
            else:
                raise RuntimeError(
                    "Expected non-reentrant activation checkpoint generator to be exhausted, but it was not!"
                )

        #  Ensure that we no longer hold on to the generator. always_call=True helps ensure we
        # clear this even in the case of exception in fwd pass.
        checkpoint.state(module)._ac_generator = None

    checkpoint.state(module).enable_hook = True
    module.register_forward_pre_hook(forward_pre_hook, with_kwargs=True)
    module.register_forward_hook(forward_hook, prepend=True, always_call=True)
    return module
```

## 逐行讲解 / What's happening

1. **`@contract(_CheckpointState)` 装饰器 (line 37)**:
   - 中文: `contract` 是 composable API 框架自带的注册机制,核心作用是把 `_CheckpointState` 这个状态对象**挂在原模块上**(通过 `checkpoint.state(module)` 拿到),而不是创建一个新模块包住它。这是整个 "composable" 的基础 idea:状态附加在模块上,模块身份不变。
   - English: `contract` is the composable-API framework's registration mechanism. Its key role is to attach a `_CheckpointState` object **onto the original module** (retrievable via `checkpoint.state(module)`) rather than creating a wrapping module. This is the foundation of "composable": state lives alongside the module, the module's identity stays intact.

2. **`forward_pre_hook` 内部的 `gen = _checkpoint_without_reentrant_generator(...)` (lines 101-110)**:
   - 中文: 这一行最关键——它没有真的"做" checkpoint,而是创建了一个**generator 对象**。这个 generator 内部大概长这样:`# pre-stuff; yield; # post-stuff`。也就是说,checkpoint 的"前期准备"和"后期清理"被串到一根 generator 上,等着外部用两次 `next()` 来推进。
   - English: This is the crux — the call doesn't *do* the checkpoint, it creates a **generator object**. Inside that generator the logic looks roughly like `# pre-stuff; yield; # post-stuff`. The "before" and "after" halves of checkpointing are stitched into one generator, waiting for two external `next()` calls to drive it.

3. **`next(gen)` 在 pre-hook 里 (line 112)**:
   - 中文: 第一次 `next()` 把 generator 推进到 `yield` 那一行,执行所有前期工作——保存输入、注册 saved-tensor hook 把激活"扔掉"以备反向重算等等。然后就停在 `yield` 处。注意此时 `module.forward(*args)` **还没运行**,因为我们在 pre-hook 里。
   - English: The first `next()` advances the generator up to `yield`, running all the pre-work — stashing inputs, registering saved-tensor hooks that *drop* activations for later recompute, etc. The generator then parks at `yield`. Crucially, `module.forward(*args)` **hasn't run yet** — we're still in the pre-hook.

4. **`next(gen)` 在 post-hook 里 (line 120)**:
   - 中文: 现在 forward 跑完了,post-hook 触发,再调一次 `next(gen)` 推进到 generator 末尾——会立刻抛出 `StopIteration`,因为 `yield` 只有一个。我们 `except StopIteration: pass`,这正是"正确退出上下文"的信号。**关键**:`else:` 分支(没抛 StopIteration)反而是错的,所以代码主动 `raise RuntimeError` ——意思是"你这个 generator 怎么还能再 yield 一次?"
   - English: Forward has completed; the post-hook fires. A second `next(gen)` advances the generator past its single `yield`, which immediately raises `StopIteration`. We `except StopIteration: pass` — that's the *expected* signal that the context exited cleanly. **Subtle**: the `else:` branch (no `StopIteration` thrown) is actually the error case — `raise RuntimeError` means "your generator somehow yielded again, that's a bug."

5. **`always_call=True` on `register_forward_hook` (line 134)**:
   - 中文: 这是新参数。意思是即使 forward 抛异常,post-hook 也会被调用——保证 generator 一定被推到 `StopIteration`,防止泄漏。结合最后那行 `_ac_generator = None`,确保不会在异常路径上挂着一个半 yield 的 generator。
   - English: A relatively new hook argument. It guarantees the post-hook runs even if forward raises — ensuring the generator always reaches `StopIteration` and we don't leak a half-yielded generator on the exception path. Combined with the final `_ac_generator = None`, no half-state survives an exception.

6. **`_no_hook` context manager (lines 16-29)**:
   - 中文: 反向重算时,`forward` 会被再调一次,如果这时 hook 还开着,会再次触发 checkpoint——递归无限循环。`_no_hook` 在重算期间把 `enable_hook` 标志拨成 `False`,跳过 hook 逻辑;`finally` 块保证恢复。
   - English: During backward, `forward` is re-invoked for recomputation; if hooks are still active you'd recursively re-checkpoint — infinite loop. `_no_hook` flips `enable_hook` off during the recompute window; the `finally` clause restores it. A classic "disable myself while I'm running" guard.

7. **`prepend=True` (line 134)**:
   - 中文: 把这个 hook 放在所有其他 forward hook 的最前面,这样 checkpoint 的"包装"语义在其他 hook 之前完成——比如其他 hook 想看 output,会看到一个已经经过 checkpoint 处理的 output。
   - English: Places this hook at the front of the forward-hook list, so checkpoint's "wrap-around" semantics happen before other hooks — for example any other hook reading `output` sees the post-checkpoint output.

## 类比 / The analogy

想象你在录广播节目,需要在嘉宾说话前后各播一段音乐——传统做法是把每段嘉宾发言剪好,前面接一段、后面接一段,工作量很大(对应"改 forward")。Wrapper 做法是把嘉宾雇成另一个人("`CheckpointWrapper(嘉宾)`"),所有访客名单里都写"嘉宾的代理人"(FQN 被污染)。Composable 的做法呢?你雇一位"音控师",在嘉宾进话筒前按一个键(pre-hook,`next(gen)` 起音乐),嘉宾说完离开话筒时再按一个键(post-hook,`next(gen)` 停音乐)。嘉宾本人没换,流程也不变,音控师只在两个时刻按了两下键——而这两下"按键"其实推进的是同一个 generator 的两次 `next()`,所以"按错顺序"或"漏按"是物理上不可能的。

Picture recording a radio show where each guest gets a music cue before and after they speak. The traditional approach is to re-edit every guest's audio with music spliced on either side — lots of work (analogous to rewriting forward). The wrapper approach replaces the guest with a "stunt double" wearing the guest's name tag — the guest list now reads `guest_proxy` everywhere (FQN pollution). The composable approach hires a sound engineer who hits one button before the guest steps to the mic (pre-hook: `next(gen)` starts the music) and another button when they finish (post-hook: `next(gen)` stops it). The guest is untouched, the show flow is untouched, and because both button-presses drive the same single generator's `next()`, getting the order wrong or skipping one is structurally impossible.

## 自己跑一遍 / Try it yourself

```python
# try_gen_ctx.py — see the generator trick in isolation, no PyTorch needed.
import contextlib

def two_phase_ctx(name):
    """A generator that's a two-phase context manager."""
    print(f"  [{name}] enter (pre-hook would call next() here)")
    yield
    print(f"  [{name}] exit  (post-hook would call next() here)")

print("Simulated forward of module A then module B (A wraps B):")

gen_A = two_phase_ctx("A")
next(gen_A)              # pre-hook A
gen_B = two_phase_ctx("B")
next(gen_B)              # pre-hook B
print("    >>> forward of B runs here")
try: next(gen_B)         # post-hook B
except StopIteration: print("  [B] StopIteration caught (expected)")
print("    >>> forward of A continues here")
try: next(gen_A)         # post-hook A
except StopIteration: print("  [A] StopIteration caught (expected)")
```

运行 / Run with:
```bash
python try_gen_ctx.py     # pure stdlib, no install needed
```

预期输出 / Expected output:
```
Simulated forward of module A then module B (A wraps B):
  [A] enter (pre-hook would call next() here)
  [B] enter (pre-hook would call next() here)
    >>> forward of B runs here
  [B] exit  (post-hook would call next() here)
  [B] StopIteration caught (expected)
    >>> forward of A continues here
  [A] exit  (post-hook would call next() here)
  [A] StopIteration caught (expected)
```

注意 enter / exit 的嵌套顺序正好是栈式的:A 先 enter → B enter → B exit → A exit。这正是 PyTorch 在嵌套 `checkpoint` 多层模块时利用的性质——每个模块的 generator 是独立对象,挂在各自模块的 `_ac_generator` 字段上,互不干扰。

The enter/exit ordering is stack-like: A-enter → B-enter → B-exit → A-exit. That's exactly the property PyTorch leverages when you `checkpoint` multiple nested modules — each module's generator is its own object, stashed in its own `_ac_generator` field, so they never interfere.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **`contextlib.contextmanager`** / **`contextlib.contextmanager`**: 这就是 generator-as-context-manager 的标准用法,Python 标准库自带 / The canonical stdlib version of generator-as-context-manager. The composable API uses the same primitive manually.
- **FSDP2 `fully_shard`** / **FSDP2 `fully_shard`**: 同样用 `@contract` + hooks 注册,不改 FQN,可以与 `checkpoint(...)` 任意组合 / Uses the same `@contract` + hooks registration, leaves FQN intact, and *composes* freely with `checkpoint(...)` on the same module.
- **PyTorch profiler `record_function`** / **PyTorch profiler `record_function`**: 也是 pre-hook 打 tag、post-hook 结束 tag 的模式 / Same "pre-hook tags, post-hook closes tag" pattern.
- **JAX `custom_vjp` with `@checkpoint`** / **JAX `jax.checkpoint`**: JAX 的对应物,实现思路完全不同(它走 trace + rematerialization),但 user-facing 行为类似 / JAX's analogue uses trace-rewriting instead of hooks; user-facing behavior is similar.
- **`pytest` 的 `yield`-based fixture** / **`pytest` `yield`-based fixtures**: setup 写 `yield` 之前,teardown 写 `yield` 之后——一模一样的 generator-context 套路 / Setup before `yield`, teardown after — the exact same generator-as-bracketing-context pattern.

## 注意事项 / Caveats / when it breaks

- **`use_reentrant=True` 不支持** / **`use_reentrant=True` is unsupported**: 老的 reentrant 实现依赖于"forward 是一段可重入的函数",但 composable API 是 hook-based 的,没法重入 / The old reentrant implementation assumes "forward is a re-invocable function," which doesn't translate to a hook-based API.
- **不能在 generator 抛 `StopIteration` 之外的异常时静默吞掉** / **Don't swallow exceptions other than `StopIteration`**: 代码只在 `except StopIteration: pass`;其他异常会冒泡。这是对的 / Code only catches `StopIteration`; other exceptions propagate. That's correct.
- **`always_call=True` 必须打开** / **`always_call=True` is mandatory**: 否则 forward 抛异常时 post-hook 不会被调用,generator 留半步——下一次 forward 会因 `_ac_generator is not None` 状态不对而出错 / Otherwise a forward exception leaves the generator half-advanced; the next forward call hits a stale `_ac_generator` and breaks.
- **不能 `checkpoint(...)` 同一个模块两次** / **Don't `checkpoint(...)` the same module twice**: 会注册两套 hooks,`_ac_generator` 字段会被覆盖,行为未定义。`@contract` 装饰器其实会拒绝重复注册,但记得别这么用 / It would install two hook pairs and overwrite the shared `_ac_generator` slot — undefined behavior. The `@contract` decorator does refuse re-registration, but treat it as an invariant.
- **`enable_hook` 字段必须共享** / **`enable_hook` must be shared, not copied**: 因为反向重算时 `_no_hook` 要能把它拨成 `False`;如果你深拷贝了模块,新模块的 `_ac_generator` 是另一个独立状态 / Backward recomputation relies on `_no_hook` mutating the shared flag; if you deep-copy the module, the copy gets its own independent state and behavior gets weird.

## 延伸阅读 / Further reading

- [PyTorch RFC: Composable Distributed Primitives](https://github.com/pytorch/pytorch/issues/93268) — 整个 `_composable` 命名空间的设计动机
- [PEP 343 — The `with` Statement](https://peps.python.org/pep-0343/) — `contextlib.contextmanager` 背后的数学,理解了它再看本节代码会很顺
- [`torch.utils.checkpoint` source — `_checkpoint_without_reentrant_generator`](https://github.com/pytorch/pytorch/blob/main/torch/utils/checkpoint.py) — 这里 `next(gen)` 真正驱动的那个 generator 的源码
- [FSDP2 design note](https://github.com/pytorch/pytorch/blob/main/torch/distributed/_composable/fsdp/fully_shard.py) — 用同一套 `@contract` 模板,可以对照看 composable 模式如何复用
