# Decision 002: Strict portability & branch isolation

Date: 2026-06-15
Status: ✅ accepted (hard constraint from user)

## Context

User 当前用 daily-code 仓库的 `nanowam` 分支做这个工程,**但后续会迁到一个新的独立仓库**(Claude 目前只能访问 daily-code,所以暂时寄存这里)。

同时 daily-code 主线在按 daily-code skill 自动生产教学笔记并 squash-merge 到 main,**不能被 nanoWAM 的工程线污染**。

## Decision

### 分支隔离

- `nanowam` 分支永远不 PR、不 merge 进 `main`
- Claude 不许调 GitHub MCP 的 `create_pull_request` / `merge_pull_request` 把 nanowam 的内容推上 main
- 会话结束只在 `nanowam` 分支 `git push origin nanowam`,daily-code 主线的所有自动化(daily-code skill 那套合 main 的逻辑)对这个分支不适用

### 子目录自包含

`nanowam/` 子目录必须是一个完整、可直接迁出的项目:

- 不引用 `nanowam/` 之外的任何文件
- 不写 `../2026/...`、`../INDEX.md`、`../.config/...` 这类相对路径
- 所有依赖 / 配置 / 文档 / 代码全在 `nanowam/` 内
- 文档里给参考代码贴的是 `/tmp/daily_code_cache/<repo>/<path>` 这种 **user 本地的外部 clone 路径**,**不是** daily-code 仓库的内部路径

### 迁仓判定

判定标准:
```bash
cp -r nanowam/ /tmp/new-repo/
cd /tmp/new-repo
# 应该一切照常,不会因为找不到 daily-code 主仓库里某个文件而崩
```

迁仓后只需要:
1. 在新仓库里把 `nanowam/` 顶层文件(README、PROGRESS、所有 stage 目录)平移到新仓库根目录
2. 不需要 fix 任何 import 路径或文档链接

## Rationale

- daily-code 主线是日更教学笔记,nanoWAM 是长期工程项目,**生命周期完全不同**,merge 进 main 会让 daily-code 的 git history 变脏、也会让以后迁仓时的 git history 不干净
- 自包含让"迁仓"动作变成纯 `cp -r`,不需要任何手工修补
- Claude 任何时候都不需要"懂"daily-code 主线在做什么,只需要懂 nanoWAM 这个子项目

## Enforcement

- 本约束已写进 `PROGRESS.md` 顶部的 "HARD RULES" 节,任何新 Claude 上来必读
- 任何会话结束前 Claude **必须自检**:
  - 没有调过 `create_pull_request` / `merge_pull_request`
  - 所有改动都在 `nanowam/` 子目录内
  - 没有在文档里出现 `../2026/`、`../INDEX.md` 等仓库内部相对路径

## References

- daily-code 主线的合并自动化:`.claude/skills/daily-code-teach/SKILL.md` (Step 4),**不适用** nanowam
- daily-code repo CLAUDE.md 里的 git rule:"通过 PR + squash-merge 进 main",**只对 daily code skill 适用**,不对 nanowam 适用
