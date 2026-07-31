---
date: 2026-07-31
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/nn/modules/module.py
permalink: https://github.com/pytorch/pytorch/blob/df5343c309835f328617b748dcb1a101de712f3c/torch/nn/modules/module.py#L1776-L1917
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, pytorch, module-hooks, forward]
---

# PyTorch _call_impl：一次 forward 要穿过 hook 门厅 / PyTorch _call_impl: A Forward Pass Walks Through the Hook Lobby

> **一句话 / In one line**: `module(...)` 不是直接调用 `forward`，它先处理 pre-hook、backward hook、forward hook 和异常时的 `always_call` hook。 / `module(...)` does not simply call `forward`; it routes through pre-hooks, backward hooks, forward hooks, and exception-time `always_call` hooks.

## 为什么重要 / Why this matters

调试器、profiling、activation capture、LoRA 注入和监控逻辑都喜欢挂 hook。理解 `_call_impl` 能解释两个常见现象：没有 hook 时 PyTorch 很快走 fast path；一旦有 hook，输入和输出都有可能被用户代码改写。

Debuggers, profilers, activation capture, LoRA injection, and monitoring code often attach hooks. Understanding `_call_impl` explains two common behaviors: PyTorch takes a fast path when there are no hooks, and once hooks exist, both inputs and outputs can be rewritten by user code.

## 代码 / The code

`pytorch/pytorch` — [`torch/nn/modules/module.py`](https://github.com/pytorch/pytorch/blob/df5343c309835f328617b748dcb1a101de712f3c/torch/nn/modules/module.py#L1776-L1917)

```python
def _wrapped_call_impl(self, *args, **kwargs):
    if self._compiled_call_impl is not None:
        return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
    else:
        return self._call_impl(*args, **kwargs)

def _call_impl(self, *args, **kwargs):
    forward_call = (self._slow_forward if torch._C._get_tracing_state() else self.forward)
    if not (self._backward_hooks or self._backward_pre_hooks or self._forward_hooks or self._forward_pre_hooks
            or _global_backward_pre_hooks or _global_backward_hooks
            or _global_forward_hooks or _global_forward_pre_hooks):
        return forward_call(*args, **kwargs)

    result = None
    called_always_called_hooks = set()

    def inner():
        nonlocal result, args, kwargs
        full_backward_hooks, non_full_backward_hooks = [], []
        backward_pre_hooks = []
        if self._backward_pre_hooks or _global_backward_pre_hooks:
            backward_pre_hooks = self._get_backward_pre_hooks()
        if self._backward_hooks or _global_backward_hooks:
            full_backward_hooks, non_full_backward_hooks = self._get_backward_hooks()
        if _global_forward_pre_hooks or self._forward_pre_hooks:
            for hook_id, hook in (*_global_forward_pre_hooks.items(), *self._forward_pre_hooks.items()):
                if hook_id in self._forward_pre_hooks_with_kwargs:
                    args_kwargs_result = hook(self, args, kwargs)
                    if args_kwargs_result is not None:
                        if isinstance(args_kwargs_result, tuple) and len(args_kwargs_result) == 2:
                            args, kwargs = args_kwargs_result
                        else:
                            raise RuntimeError("forward pre-hook must return None or a tuple of (new_args, new_kwargs)")
                else:
                    args_result = hook(self, args)
                    if args_result is not None:
                        if not isinstance(args_result, tuple):
                            args_result = (args_result,)
                        args = args_result

        bw_hook = None
        if full_backward_hooks or backward_pre_hooks:
            bw_hook = BackwardHook(self, full_backward_hooks, backward_pre_hooks)
            args = bw_hook.setup_input_hook(args)

        result = forward_call(*args, **kwargs)
        if _global_forward_hooks or self._forward_hooks:
            for hook_id, hook in (*_global_forward_hooks.items(), *self._forward_hooks.items()):
                if hook_id in self._forward_hooks_always_called or hook_id in _global_forward_hooks_always_called:
                    called_always_called_hooks.add(hook_id)
                if hook_id in self._forward_hooks_with_kwargs or hook_id in _global_forward_hooks_with_kwargs:
                    hook_result = hook(self, args, kwargs, result)
                else:
                    hook_result = hook(self, args, result)
                if hook_result is not None:
                    result = hook_result
        if bw_hook:
            result = bw_hook.setup_output_hook(result)
        return result

    if torch.compiler.is_compiling():
        return inner()
    try:
        return inner()
    except Exception:
        for hook_id, hook in self._forward_hooks.items():
            if hook_id in self._forward_hooks_always_called and hook_id not in called_always_called_hooks:
                try:
                    hook_result = hook(self, args, result)
                    if hook_result is not None:
                        result = hook_result
                except Exception as e:
                    warnings.warn("module forward hook with ``always_call=True`` raised an exception "
                                  f"that was silenced as another error was raised in forward: {str(e)}", stacklevel=2)
                    continue
        raise
```

## 逐行讲解 / What's happening

1. **第 1776-1780 行 / Lines 1776-1780 (`_compiled_call_impl`)**:
   - 中文: 如果模块已经被编译，调用会先进编译后的入口。
   - English: If the module has a compiled call implementation, execution enters that path first.
2. **第 1785-1791 行 / Lines 1785-1791 (fast path)**:
   - 中文: 没有任何本地或全局 hook 时，直接调用 `forward`，避免额外调度成本。
   - English: With no local or global hooks, PyTorch calls `forward` directly and avoids dispatch overhead.
3. **第 1807-1827 行 / Lines 1807-1827 (pre-hooks)**:
   - 中文: forward pre-hook 可以返回新参数；带 kwargs 的 hook 必须返回 `(args, kwargs)`。
   - English: Forward pre-hooks can return new arguments; hooks with kwargs must return `(args, kwargs)`.
4. **第 1834-1850 行 / Lines 1834-1850 (forward + post-hooks)**:
   - 中文: 真正的 `forward` 在中间发生；forward hook 如果返回非 `None`，会替换输出。
   - English: The real `forward` happens in the middle; a forward hook that returns non-`None` replaces the output.
5. **第 1885-1916 行 / Lines 1885-1916 (exception path)**:
   - 中文: `always_call=True` 的 hook 即使 forward 抛错也要尽量执行，通常用于清理或记录。
   - English: Hooks marked `always_call=True` are attempted even when `forward` raises, usually for cleanup or logging.

## 类比 / The analogy

这像机场安检通道：没托运行李、没特殊检查时直接过；一旦有额外规则，就要经过改签、检查、盖章、异常处理这些窗口。

It is like an airport security lane: with no special baggage or checks, you pass straight through; with extra rules, you visit the reroute, inspection, stamp, and incident desks.

## 自己跑一遍 / Try it yourself

```python
def call(forward, x, pre_hooks=(), post_hooks=()):
    args = (x,)
    for h in pre_hooks:
        new_args = h(args)
        if new_args is not None:
            args = new_args if isinstance(new_args, tuple) else (new_args,)
    result = forward(*args)
    for h in post_hooks:
        new_result = h(args, result)
        if new_result is not None:
            result = new_result
    return result

print(call(lambda x: x * 2, 3))
print(call(lambda x: x * 2, 3, pre_hooks=[lambda a: (a[0] + 1,)], post_hooks=[lambda a, r: r + 10]))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
6
18
```

第二次调用里，pre-hook 先把输入从 `3` 改成 `4`，post-hook 再把输出从 `8` 改成 `18`。

In the second call, the pre-hook changes the input from `3` to `4`, then the post-hook changes the output from `8` to `18`.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Activation capture** / **Activation capture**: forward hook 常用来保存中间层输出。 / Forward hooks are often used to save intermediate activations.
- **Instrumentation** / **Instrumentation**: profiler 或监控代码会用 `always_call` 记录失败请求。 / Profilers or monitoring code use `always_call` to record failed calls.

## 注意事项 / Caveats / when it breaks

- **hook 会改变语义** / **Hooks can change semantics**: pre-hook 和 forward hook 都能改输入输出，调 bug 时要先列出已注册 hook。 / Pre-hooks and forward hooks can rewrite inputs and outputs; list registered hooks first when debugging.
- **编译路径更严格** / **Compiled paths are stricter**: 源码注释也说明异常后再跑 hook 与编译并不完全等价。 / The source comment notes that rerunning hooks after exceptions is not fully equivalent under compilation.

## 延伸阅读 / Further reading

- [PyTorch Module call source permalink](https://github.com/pytorch/pytorch/blob/df5343c309835f328617b748dcb1a101de712f3c/torch/nn/modules/module.py#L1776-L1917)
