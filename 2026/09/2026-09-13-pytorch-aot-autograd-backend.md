---
date: 2026-09-13
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/_dynamo/backends/common.py
permalink: https://github.com/pytorch/pytorch/blob/ea89e4e90dc68302e8ef5cba3ddbdaa9d50d9512/torch/_dynamo/backends/common.py#L74-L167
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, pytorch, torch-compile, aot-autograd, compiler-backend, graph-inputs]
---

# PyTorch AOT Autograd：把 backward 编译器包在禁追踪壳里 / PyTorch AOT Autograd: Keep the Backward Compiler Outside Dynamo

> **一句话 / In one line**: AOT Autograd 不只编译 forward，它还把 backward compiler 和生成出的 backward graph 明确排除在 Dynamo 追踪之外。 / AOT Autograd compiles more than forward; it explicitly keeps both the backward compiler and its generated backward graph outside Dynamo tracing.

## 为什么重要 / Why this matters

中文：`torch.compile` 的后端本身也是 Python 代码。如果 Dynamo 又去追踪后端，编译“编译器”的过程会递归扩大，日志、计数器和错误边界都会变得难以理解。`AotAutograd` 在交给 `aot_module_simplified` 前包装 backward compiler，在返回 compiled graph 后再包一次，形成两个不同的禁追踪边界。

English: A `torch.compile` backend is itself Python code. If Dynamo traces the backend, compiling the compiler can recursively expand the graph and blur logging, counters, and failure boundaries. `AotAutograd` wraps the backward compiler before calling `aot_module_simplified`, then disables tracing on the returned compiled graph as well.

## 代码 / The code

`pytorch/pytorch` — [`torch/_dynamo/backends/common.py`](https://github.com/pytorch/pytorch/blob/ea89e4e90dc68302e8ef5cba3ddbdaa9d50d9512/torch/_dynamo/backends/common.py#L74-L167)

```python
class AotAutograd:
    def __init__(self, **kwargs: Unpack[AotAutogradKwargs]) -> None:
        self.__name__ = "compiler_fn"
        self.kwargs: AotAutogradKwargs = kwargs

    @property
    def _dynamo_backend_init(self) -> Any | None:
        return getattr(self.kwargs.get("fw_compiler"), "_dynamo_backend_init", None)

    def __call__(
        self, gm: torch.fx.GraphModule, example_inputs: Sequence[Any], **kwargs: Any
    ) -> Callable[..., Any]:
        if kwargs:
            log.warning("aot_autograd-based backend ignoring extra kwargs %s", kwargs)

        if any(isinstance(x, (list, tuple, dict)) for x in example_inputs):
            return flatten_graph_inputs(
                gm,
                example_inputs,
                self,
            )

        decompositions = self.kwargs.get("decompositions")
        if callable(decompositions):
            self.kwargs["decompositions"] = decompositions()

        counters["aot_autograd"]["total"] += 1

        def wrap_bw_compiler(bw_compiler_fn: Callable[P, R]) -> Callable[..., R]:
            def _wrapped_bw_compiler(*args: P.args, **kwargs: P.kwargs) -> R:
                # Stop Dynamo from tracing both the compiler and its output.
                return disable(
                    disable(
                        bw_compiler_fn, reason="do not trace backward compiler function"
                    )(*args, **kwargs),
                    reason="do not trace generated backwards pass",
                )

            _wrapped_bw_compiler._is_wrapped_bw_compiler = True
            return _wrapped_bw_compiler

        bw_compiler = self.kwargs.get("bw_compiler") or self.kwargs["fw_compiler"]

        if isinstance(bw_compiler, SerializableAOTDispatchCompiler):
            if not getattr(bw_compiler.compiler_fn, "_is_wrapped_bw_compiler", False):
                bw_compiler.compiler_fn = wrap_bw_compiler(bw_compiler.compiler_fn)
        elif not getattr(bw_compiler, "_is_wrapped_bw_compiler", False):
            bw_compiler = wrap_bw_compiler(bw_compiler)

        self.kwargs["bw_compiler"] = bw_compiler
        self.kwargs["inference_compiler"] = (
            self.kwargs.get("inference_compiler") or self.kwargs["fw_compiler"]
        )

        from functorch.compile import nop
        from torch._inductor.debug import enable_aot_logging

        if self.kwargs.get("fw_compiler", None) is nop:
            patch_config = patch("functorch.compile.config.debug_assert", True)
        else:
            patch_config = contextlib.nullcontext()

        try:
            with enable_aot_logging(), patch_config:
                cg = aot_module_simplified(
                    gm,
                    example_inputs,
                    **self.kwargs,
                )
                counters["aot_autograd"]["ok"] += 1
                return disable(cg, reason="do not trace AOT-compiled graph")
        except TensorifyScalarRestartAnalysis:
            raise
        except Exception:
            counters["aot_autograd"]["not_ok"] += 1
            raise
```

## 逐行讲解 / What's happening

1. **第 85-96 行 / Lines 85-96 (input shape boundary)**:
   - 中文：如果 example input 里还有 list、tuple 或 dict，先交给 `flatten_graph_inputs`。AOT 编译器拿到的是稳定的扁平签名，原始容器结构由包装层恢复。
   - English: Structured inputs are handed to `flatten_graph_inputs` first. The AOT compiler receives a stable flat signature, while the wrapper restores the original container structure.
2. **第 98-105 行 / Lines 98-105 (decomposition and accounting)**:
   - 中文：可调用的 decomposition thunk 在“真正编译时”才解析；`total` 计数器放在成功与失败都能到达的位置。
   - English: A callable decomposition thunk is resolved at compile time, and the `total` counter is incremented before success or failure.
3. **第 106-123 行 / Lines 106-123 (two disables)**:
   - 中文：第一层 `disable` 包住 compiler function，第二层包住 compiler 的返回值。它们分别阻止追踪后端本身和后端生成的 backward。
   - English: The first `disable` covers the compiler function; the second covers the compiler's output. They prevent tracing the backend and the generated backward independently.
4. **第 125-136 行 / Lines 125-136 (compiler selection)**:
   - 中文：没有独立 backward compiler 时，forward compiler 兜底；inference compiler 也有同样的 fallback，保证配置可以只提供最小集合。
   - English: The forward compiler is the fallback for backward and inference, so a minimal configuration does not need three separate functions.
5. **第 150-167 行 / Lines 150-167 (compile boundary)**:
   - 中文：`aot_module_simplified` 是真正的 AOT 入口。成功增加 `ok` 并禁追踪返回 graph；失败增加 `not_ok`，但让原始异常继续向上传播。
   - English: `aot_module_simplified` is the actual AOT entry point. Success increments `ok` and disables tracing on the returned graph; failure increments `not_ok` and preserves the original exception.

## 类比 / The analogy

中文：像一家工厂的总装线。Dynamo 负责把用户程序送进工厂；AOT Autograd 是负责安排 forward/backward 两条生产线的调度员。调度员可以被叫到，但不能把自己再当成产品送回总装线。

English: Think of an assembly plant. Dynamo sends the user's program into the factory; AOT Autograd schedules separate forward and backward production lines. The scheduler can be invoked, but it must not become another product sent down the assembly line.

## 自己跑一遍 / Try it yourself

```python
def disable(fn, reason):
    def wrapped(*args, **kwargs):
        return fn(*args, **kwargs)
    wrapped.reason = reason
    return wrapped


def wrap_backward(compiler):
    def wrapped(*args, **kwargs):
        generated = compiler(*args, **kwargs)
        return disable(generated, "do not trace generated backwards pass")
    return disable(wrapped, "do not trace backward compiler function")


calls = []
def compiler(graph):
    calls.append(graph)
    return lambda x: x + 1

backward = wrap_backward(compiler)
print(backward("graph")(4), len(calls), backward.reason)
```

运行 / Run with:

```bash
python try.py
```

预期输出 / Expected output:

```text
5 1 do not trace backward compiler function
```

中文：这里没有真的调用 Dynamo，但能看到两个边界：compiler 只执行一次，返回的 callable 也被标记为“不要再次追踪”。真实实现还要处理 FX graph、autograd partition 和 backend counters。

English: This toy does not invoke Dynamo, but it exposes the two boundaries: the compiler runs once, and its returned callable is marked not to be traced again. The real implementation adds FX graphs, autograd partitioning, and backend counters.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **TorchInductor backends** / **TorchInductor backends**: 编译器通常需要把内部 helper 与用户 graph 分开。 / Compilers commonly separate internal helpers from the user graph.
- **functorch partitioners** / **functorch partitioners**: forward 和 backward 可以由不同 compiler 处理。 / Forward and backward can be handled by different compiler functions.
- **callback-based compilers** / **callback-based compilers**: 回调的返回值往往还需要一个新的生命周期或 tracing contract。 / A callback's return value often needs its own lifecycle or tracing contract.

## 注意事项 / Caveats / when it breaks

- **结构化输入不能直接假设扁平** / **Do not assume inputs are flat**: list、dict 和 tuple 会改变 graph signature，必须先统一边界。
- **wrapped 标记是幂等性的一部分** / **The wrapped marker supports idempotence**: 没有 `_is_wrapped_bw_compiler` 检查，重复构造 backend 可能把多层 wrapper 套起来。
- **不要吞掉编译异常** / **Do not swallow compile failures**: 计数器用于观测，异常仍应保留原始 traceback。
- **debug assert 有成本** / **Debug asserts cost compile time**: 当前只在 `nop` 路径默认打开，是调试准确性和编译速度之间的明确取舍。

## 延伸阅读 / Further reading

- [PyTorch AOT Autograd backend](https://github.com/pytorch/pytorch/blob/ea89e4e90dc68302e8ef5cba3ddbdaa9d50d9512/torch/_dynamo/backends/common.py)
- [torch.compile programming model](https://pytorch.org/docs/stable/torch.compiler.html)
