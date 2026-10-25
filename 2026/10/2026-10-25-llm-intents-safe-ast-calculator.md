---
date: 2026-10-25
topic: diffusion
source: trending
repo: skye-harris/llm_intents
file: custom_components/llm_intents/calculator.py
permalink: https://github.com/skye-harris/llm_intents/blob/112aed155c90bdc5f9255bf1ffdd696b741a0d48/custom_components/llm_intents/calculator.py#L40-L102
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, trending, llm-tools, ast, safety]
---

# llm_intents 安全计算器：解析 AST，不执行字符串 / llm_intents Safe Calculator: Parse an AST, Never Execute a String

> **一句话 / In one line**: 这个 Home Assistant LLM 工具只解释数字 AST 的白名单节点，并给幂运算加资源上限。 / This Home Assistant LLM tool interprets only allowlisted numeric AST nodes and bounds exponentiation resources.

## 为什么重要 / Why this matters

把 LLM 的工具输入直接交给 `eval` 等于让自然语言系统获得任意代码执行入口。该实现先将表达式解析为 AST，只递归接受常量、一元运算和二元运算，并把最容易造成资源耗尽的幂运算单独看守。

Passing LLM tool input directly to `eval` creates an arbitrary-code-execution boundary. This implementation parses an AST first, recursively accepts only constants, unary operators, and binary operators, then gives the resource-heavy power operator its own guard.

## 代码 / The code

`skye-harris/llm_intents` — [`calculator.py`](https://github.com/skye-harris/llm_intents/blob/112aed155c90bdc5f9255bf1ffdd696b741a0d48/custom_components/llm_intents/calculator.py#L40-L102)

```python
def _safe_eval(expression: str) -> float:
    tree = ast.parse(expression, mode="eval")
    return _eval_node(tree.body)

def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)):
            raise TypeError(f"Unsupported constant: {node.value!r}")
        return float(node.value)
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type is ast.Pow:
            return _eval_pow(node)
        if op_type not in _SAFE_BIN_OPS:
            raise ValueError(f"Unsupported operator: {op_type.__name__}")
        return _SAFE_BIN_OPS[op_type](_eval_node(node.left), _eval_node(node.right))
    raise ValueError(f"Unsupported expression element: {type(node).__name__}")
```

## 逐行讲解 / What's happening

1. **第 47-48 行 / Lines 47-48**：`ast.parse(..., mode="eval")` 只接受表达式语法，随后代码自己决定允许哪些节点。 / `ast.parse(..., mode="eval")` accepts expression syntax only; the code then decides which nodes are allowed.
2. **第 53-57 行 / Lines 53-57**：常量还要是数值，字符串和其他 Python literal 都不能混入计算。 / Constants must still be numeric, excluding strings and other Python literals.
3. **第 70-71 与 85-102 行 / Lines 70-71 and 85-102**：幂运算绕开通用 dispatch，检查底数、指数和结果大小。 / Power bypasses general dispatch so base, exponent, and result size can be checked.

## 类比 / The analogy

像机场安检：不是让任何人进门后再观察行为，而是只放行名单上的物品类别。 / It is airport screening: rather than letting anything in and observing later, it admits only listed categories of items.

## 自己跑一遍 / Try it yourself

```python
import ast

def safe(expr):
    node = ast.parse(expr, mode="eval").body
    if not isinstance(node, ast.BinOp) or not all(isinstance(x, ast.Constant) for x in (node.left, node.right)):
        raise ValueError("numbers with one binary op only")
    return {ast.Add: lambda a,b: a+b, ast.Mult: lambda a,b: a*b}[type(node.op)](node.left.value, node.right.value)
print(safe("6 * 7"))
```

运行 / Run with: `python try.py`

预期输出 / Expected output: `42`。

示例刻意更窄；生产代码的白名单可以增长，但默认应拒绝。 / The example is deliberately narrower; a production allowlist may grow, but denial should remain the default.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **SQL 参数化 / Parameterized SQL**：数据和可执行语法分开处理。 / It separates data from executable syntax.
- **工具 schema 校验 / Tool-schema validation**：模型输出先通过一个受限接口。 / Model output first passes through a restricted interface.

## 注意事项 / Caveats / when it breaks

- **AST 安全不等于资源安全 / AST safety is not resource safety**：允许的运算也可能极慢，因此仍需阈值。 / Allowed operations can still be expensive, so thresholds remain necessary.
- **不要回退到 eval / Do not fall back to eval**：解析失败应当报错，而不是尝试“兼容”。 / A parse failure should error, not trigger a compatibility fallback.

## 延伸阅读 / Further reading

- [llm_intents calculator source](https://github.com/skye-harris/llm_intents/blob/112aed155c90bdc5f9255bf1ffdd696b741a0d48/custom_components/llm_intents/calculator.py)
