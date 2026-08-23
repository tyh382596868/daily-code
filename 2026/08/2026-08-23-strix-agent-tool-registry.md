---
date: 2026-08-23
topic: infrastructure
source: trending
repo: usestrix/strix
file: strix/agents/factory.py
permalink: https://github.com/usestrix/strix/blob/main/strix/agents/factory.py#L477-L626
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, infrastructure, agents, tool-registry]
---

# Strix agent factory：工具先注册，再装进 agent / Strix Agent Factory: Register Tools, Then Build the Agent

> **一句话 / In one line**: Strix 把基础工具、额外工具和生命周期工具合并成一个去重后的 toolset，再创建 root 或 child security agent。 / Strix merges base tools, extra tools, and lifecycle tools into a de-duplicated toolset before constructing root or child security agents.

## 为什么重要 / Why this matters

Agent 系统最容易失控的地方不是 prompt，而是工具边界：哪些工具能用、输出多大、什么时候算完成、root agent 和 child agent 有什么差异。Strix 的 factory 把这些边界集中在一个构建点。

The fragile part of an agent system is often not the prompt but the tool boundary: which tools are available, how large outputs can be, when a run is complete, and how root and child agents differ. Strix centralizes those decisions in one factory.

## 代码 / The code

`usestrix/strix` — [`strix/agents/factory.py`](https://github.com/usestrix/strix/blob/main/strix/agents/factory.py#L477-L626)

```python
_BASE_TOOLS: tuple[Tool, ...] = (
    think,
    load_skill,
    create_todo,
    list_todos,
    update_todo,
    mark_todo_done,
    mark_todo_pending,
    delete_todo,
    create_note,
    list_notes,
    get_note,
    update_note,
    delete_note,
    web_search,
    create_vulnerability_report,
    create_dependency_report,
    list_reports,
    get_report,
    list_requests,
    view_request,
    repeat_request,
    list_sitemap,
    view_sitemap_entry,
    scope_rules,
    view_agent_graph,
    send_message_to_agent,
    wait_for_agents,
    create_agent,
    stop_agent,
)
_EXTRA_TOOLS: list[Tool] = []

def _ensure_unique_tool_names(tools: Sequence[Tool]) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for tool in tools:
        if tool.name in seen:
            duplicates.add(tool.name)
        seen.add(tool.name)
    if duplicates:
        msg = f"Agent tools must have unique names: {sorted(duplicates)}"
        raise ValueError(msg)

def register_agent_tools(*tools: Tool) -> None:
    """Register tools for every scan agent built afterwards."""
    new_tools: list[Tool] = []
    for tool in tools:
        if tool not in _EXTRA_TOOLS and tool not in new_tools:
            new_tools.append(tool)

    _ensure_unique_tool_names([*_BASE_TOOLS, *_EXTRA_TOOLS, *new_tools, finish_scan, agent_finish])
    for tool in new_tools:
        _EXTRA_TOOLS.append(tool)
        logger.info("Registered extra agent tool: %s", getattr(tool, "name", tool))

def build_strix_agent(
    *,
    name: str = "agent",
    skills: list[str] | None = None,
    is_root: bool,
    scan_mode: str = "deep",
    is_whitebox: bool = False,
    interactive: bool = False,
    chat_completions_tools: bool = False,
    strict_tool_schemas: bool = True,
    system_prompt_context: dict[str, Any] | None = None,
    extra_tools: Sequence[Tool] | None = None,
    instructions_override: str | None = None,
) -> SandboxAgent[Any]:
    if instructions_override is not None:
        instructions = instructions_override
    else:
        instructions = render_system_prompt(
            skills=skills,
            scan_mode=scan_mode,
            is_whitebox=is_whitebox,
            is_root=is_root,
            interactive=interactive,
            system_prompt_context=system_prompt_context,
        )
    agent_tools = [*_EXTRA_TOOLS, *(extra_tools or [])]
    if interactive:
        agent_tools.append(respond_to_user)
    if is_root:
        tools: list[Tool] = [*_BASE_TOOLS, *agent_tools, finish_scan]
    else:
        tools = [*_BASE_TOOLS, *agent_tools, agent_finish]
    _ensure_unique_tool_names(tools)
    tools = [
        _with_bounded_result(_with_strictness(_with_coerced_arguments(tool), strict_tool_schemas))
        if isinstance(tool, FunctionTool)
        else tool
        for tool in tools
    ]
    return SandboxAgent(
        name=name,
        instructions=instructions,
        tools=tools,
        tool_use_behavior=_finish_tool_use_behavior,
        model=None,
        capabilities=[
            Filesystem(
                configure_tools=_make_filesystem_configurator(
                    chat_completions=chat_completions_tools,
                    strict_schemas=strict_tool_schemas,
                ),
            ),
            Shell(
                configure_tools=_make_shell_configurator(
                    chat_completions=chat_completions_tools,
                    strict_schemas=strict_tool_schemas,
                ),
            ),
        ],
    )
```

## 逐行讲解 / What's happening

1. **第 477-511 行 / Lines 477-511**:
   - 中文: `_BASE_TOOLS` 是所有 scan agent 都默认具备的能力集合，包括笔记、报告、代理图和子 agent 控制。
   - English: `_BASE_TOOLS` is the default capability set shared by scan agents: notes, reports, agent graph operations, and child-agent control.
2. **第 513-522 行 / Lines 513-522**:
   - 中文: 构建前检查 tool name 唯一，避免两个工具同名时模型调用歧义。
   - English: Tool names are checked for uniqueness before construction, avoiding ambiguous tool calls.
3. **第 525-539 行 / Lines 525-539**:
   - 中文: 外部工具可以注册到全局 `_EXTRA_TOOLS`，但重复对象会被忽略。
   - English: External tools can be registered globally, while duplicate tool objects are ignored.
4. **第 546-626 行 / Lines 546-626**:
   - 中文: root agent 拿 `finish_scan`，child agent 拿 `agent_finish`；随后统一套上参数 coercion、strict schema 和输出裁剪。
   - English: Root agents receive `finish_scan`; child agents receive `agent_finish`. Then function tools are wrapped for argument coercion, schema strictness, and bounded output.

## 类比 / The analogy

这像组建施工队。所有人都有安全帽和卷尺，特殊项目再加电焊机；总包负责人拿“交付工程”的表，分包负责人拿“完成分包”的表。开工前先检查工具名不能重复。

It is like assembling a construction crew. Everyone gets a helmet and measuring tape, special projects add a welder, the main contractor gets the final-delivery form, and subcontractors get subtask-completion forms. Before work starts, tool names must not collide.

## 自己跑一遍 / Try it yourself

```python
base = ["think", "notes", "reports"]
extra = []

def register(*tools):
    names = base + extra + list(tools) + ["finish", "child_finish"]
    dupes = {name for name in names if names.count(name) > 1}
    if dupes:
        raise ValueError(sorted(dupes))
    extra.extend(t for t in tools if t not in extra)

def build(is_root):
    lifecycle = "finish" if is_root else "child_finish"
    return base + extra + [lifecycle]

register("browser")
print(build(True))
print(build(False))
```

运行 / Run with:

```bash
python try.py
```

预期输出 / Expected output:

```text
['think', 'notes', 'reports', 'browser', 'finish']
['think', 'notes', 'reports', 'browser', 'child_finish']
```

这个例子展示了 root / child 共享大部分能力，只在生命周期出口上不同。

This example shows that root and child agents share most capabilities and differ mainly in their lifecycle exit tool.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **OpenAI Agents SDK toolsets** / **OpenAI Agents SDK toolsets**: agent 构建时显式传入 tools。 / Agents receive an explicit tool list at construction.
- **MCP server registry** / **MCP server registry**: 工具名是协议层调用入口，不能冲突。 / Tool names are protocol-level call targets and must not collide.
- **CI job permissions** / **CI job permissions**: root job 和 matrix child job 常常只差最终发布权限。 / Root jobs and matrix child jobs often differ only in final publishing permissions.

## 注意事项 / Caveats / when it breaks

- **全局注册有顺序性** / **Global registration is order-sensitive**: agent 构建之后再注册，旧 agent 不会自动获得新工具。 / Agents built before registration do not automatically gain new tools.
- **同名不同义最危险** / **Same name, different meaning is dangerous**: 模型只看到名字和 schema，很难猜出你想调用哪个。 / The model sees names and schemas, so collisions are hard to disambiguate.
- **输出裁剪影响推理** / **Output bounding affects reasoning**: 太小的输出上限会丢掉关键证据。 / Too-small output limits can remove critical evidence.

## 延伸阅读 / Further reading

- Strix agent factory: https://github.com/usestrix/strix/blob/main/strix/agents/factory.py
- Strix repository: https://github.com/usestrix/strix
