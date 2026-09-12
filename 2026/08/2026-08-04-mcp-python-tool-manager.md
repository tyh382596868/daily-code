---
date: 2026-08-04
topic: infrastructure
source: trending
repo: modelcontextprotocol/python-sdk
file: src/mcp/server/mcpserver/tools/tool_manager.py
permalink: https://github.com/modelcontextprotocol/python-sdk/blob/a4f4ccd091138771535e17191123f20b30fda68e/src/mcp/server/mcpserver/tools/tool_manager.py#L39-L87
difficulty: beginner
read_time: ~10 min
tags: [code-of-the-day, infrastructure, mcp, tools]
---

# MCP ToolManager：函数先变 Tool，再进注册表 / MCP ToolManager: Turn a Function into a Tool, Then Register It

> **一句话 / In one line**: `ToolManager` 把 Python callable 包装成 MCP `Tool`，处理重复注册，并在调用时统一做 name lookup。 / `ToolManager` wraps Python callables as MCP `Tool` objects, handles duplicate registration, and performs a single name lookup when tools are called.

## 为什么重要 / Why this matters

MCP 服务器最常见的开发体验是“写一个带类型标注的函数，然后暴露成 tool”。这段代码展示了 SDK 的核心边界：业务函数不直接塞进字典，而是先经过 `Tool.from_function` 提取名称、描述、schema 和结构化输出设置，再由 manager 统一管理生命周期。

The common MCP server experience is “write a typed Python function, expose it as a tool.” This code shows the SDK boundary: business functions are not stored directly in a dict. They first pass through `Tool.from_function`, which extracts name, description, schema, and structured-output settings, then the manager owns their lifecycle.

## 代码 / The code

`modelcontextprotocol/python-sdk` — [`src/mcp/server/mcpserver/tools/tool_manager.py`](https://github.com/modelcontextprotocol/python-sdk/blob/a4f4ccd091138771535e17191123f20b30fda68e/src/mcp/server/mcpserver/tools/tool_manager.py#L39-L87)

```python
def add_tool(
    self,
    fn: Callable[..., Any],
    name: str | None = None,
    title: str | None = None,
    description: str | None = None,
    annotations: ToolAnnotations | None = None,
    icons: list[Icon] | None = None,
    meta: dict[str, Any] | None = None,
    structured_output: bool | None = None,
) -> Tool:
    """Add a tool to the server."""
    tool = Tool.from_function(
        fn,
        name=name,
        title=title,
        description=description,
        annotations=annotations,
        icons=icons,
        meta=meta,
        structured_output=structured_output,
    )
    existing = self._tools.get(tool.name)
    if existing:
        if self.warn_on_duplicate_tools:
            logger.warning(f"Tool already exists: {tool.name}")
        return existing
    self._tools[tool.name] = tool
    return tool

def remove_tool(self, name: str) -> None:
    """Remove a tool by name."""
    if name not in self._tools:
        raise ToolError(f"Unknown tool: {name}")
    del self._tools[name]

async def call_tool(
    self,
    name: str,
    arguments: dict[str, Any],
    context: Context[LifespanContextT, RequestT],
    convert_result: bool = False,
) -> Any:
    """Call a tool by name with arguments."""
    tool = self.get_tool(name)
    if not tool:
        raise ToolError(f"Unknown tool: {name}")

    return await tool.run(arguments, context, convert_result=convert_result)
```

## 逐行讲解 / What's happening

1. **第 51-60 行 / Lines 51-60 (`Tool.from_function`)**:
   - 中文: callable 在这里被提升成协议对象，函数签名和元数据会变成 MCP 可描述的 tool。
   - English: The callable becomes a protocol object here; function signature and metadata become an MCP-describable tool.
2. **第 61-67 行 / Lines 61-67 (duplicate policy)**:
   - 中文: 重复注册不会覆盖旧工具，而是可选 warning 后返回已有 tool，避免热加载或装饰器重复执行时悄悄换实现。
   - English: Duplicate registration does not overwrite the old tool; it optionally warns and returns the existing tool, avoiding silent implementation swaps during reloads or repeated decorators.
3. **第 75-87 行 / Lines 75-87 (call boundary)**:
   - 中文: 调用路径只做 name lookup 和未知工具错误，参数验证、上下文注入和结果转换交给 `tool.run`。
   - English: The call path only performs name lookup and unknown-tool errors; argument validation, context injection, and result conversion are delegated to `tool.run`.

## 类比 / The analogy

这像剧院前台登记演员：演员本人不是直接冲上舞台，而是先拿到角色卡、出场名和道具说明。演出时调度员按角色名叫人，找不到就报错。

It is like a theater desk registering actors: the actor does not rush onto stage directly. They first get a role card, stage name, and prop notes. During the show, the dispatcher calls by role name and errors if no such role exists.

## 自己跑一遍 / Try it yourself

```python
class ToolManager:
    def __init__(self):
        self.tools = {}
    def add_tool(self, fn, name=None):
        tool_name = name or fn.__name__
        if tool_name in self.tools:
            return self.tools[tool_name]
        self.tools[tool_name] = fn
        return fn
    def call_tool(self, name, args):
        if name not in self.tools:
            raise KeyError(name)
        return self.tools[name](**args)

tm = ToolManager()
tm.add_tool(lambda x: x + 1, name="inc")
print(tm.call_tool("inc", {"x": 2}))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
3
```

中文: manager 负责注册和查找，具体参数执行仍然在 tool 自己那里。
English: The manager owns registration and lookup; actual argument execution still belongs to the tool.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **FastAPI route registration** / **FastAPI route registration**: 函数会先变成 route 对象，再进入 router。 / Functions become route objects before entering the router.
- **Click command groups** / **Click command groups**: Python 函数会包装成 command，再按名称分发。 / Python functions are wrapped as commands and dispatched by name.

## 注意事项 / Caveats / when it breaks

- **重复注册策略保守** / **Duplicate policy is conservative**: 想覆盖 tool 时，不能只再 add 一次，需要显式 remove 或换名。 / To replace a tool, adding it again is not enough; remove it explicitly or use another name.
- **调用安全在下层** / **Call safety lives below**: `ToolManager` 不解析参数 schema，安全边界主要在 `Tool.run`。 / `ToolManager` does not parse argument schemas; the safety boundary mostly lives in `Tool.run`.

## 延伸阅读 / Further reading

- MCP Python SDK repository: https://github.com/modelcontextprotocol/python-sdk
- Tool manager source: https://github.com/modelcontextprotocol/python-sdk/blob/a4f4ccd091138771535e17191123f20b30fda68e/src/mcp/server/mcpserver/tools/tool_manager.py#L39-L87
