---
date: 2026-09-22
topic: infrastructure
source: trending
repo: chunxiaoxx/nautilus-compass
file: session_search.py
permalink: https://github.com/chunxiaoxx/nautilus-compass/blob/c7c3126c78c0e92769cf263421b37fe58c3bc5c9/session_search.py#L30-L97
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, infrastructure, agent-memory, search, ranking]
---

# Nautilus Compass 会话检索：先过滤，再让关键词有层次 / Nautilus Compass Session Search: Filter First, Then Weight Keywords

> **一句话 / In one line**: Compass 用轻量 frontmatter 做过滤，用 name/description/body 三层权重打分，再按得分和新鲜度返回会话。 / Compass filters with lightweight frontmatter, scores name/description/body at different weights, then returns sessions by score and recency.

## 为什么重要 / Why this matters

agent memory 的第一版不一定需要向量数据库。这个文件展示了一条很实用的本地 fallback：先从 Markdown session 的 frontmatter 读取结构化字段，再用子串命中做快速排序。只要搜索目标是“最近的 bugfix”“某个项目的 drift 记录”，这种方法已经能提供可解释结果。

An agent memory layer does not need a vector database on day one. This file shows a practical local fallback: read structured fields from Markdown frontmatter, then rank substring matches. For queries such as “recent bugfixes” or “drift records for one project,” the result is fast and explainable.

最有意思的是 body 的平方根衰减。标题里出现一次关键词比正文里出现十次更有价值，但正文重复仍然会增加分数，只是边际收益越来越小。它是一个很小的 ranking heuristic，却把“信号位置”和“重复噪声”区分开了。

The most interesting detail is the square-root decay for body hits. One match in a title is more valuable than ten repeated matches in the body, but body repetition still contributes with diminishing returns. It is a tiny ranking heuristic that separates signal location from repetition noise.

## 代码 / The code

`chunxiaoxx/nautilus-compass` — [`session_search.py`](https://github.com/chunxiaoxx/nautilus-compass/blob/c7c3126c78c0e92769cf263421b37fe58c3bc5c9/session_search.py#L30-L97)

```python
def parse_fm_simple(text: str) -> dict:
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 4)
    if end < 0:
        return {}
    fm: dict = {}
    for line in text[4:end].splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            fm[k.strip()] = v.strip().strip('"').strip("'")
    return fm


def score_session(text: str, fm: dict, query_terms: list[str]) -> float:
    """Token-overlap score · weighted: name 3× · description 2× · body 1×."""
    name = fm.get("name", "").lower()
    desc = fm.get("description", "").lower()
    body = text[len(text) // 4:].lower()
    score = 0.0
    for t in query_terms:
        t = t.lower()
        if not t:
            continue
        score += 3.0 * name.count(t)
        score += 2.0 * desc.count(t)
        score += 1.0 * (body.count(t) ** 0.5)
    return score


def search(query: str, drift: str | None = None, type_filter: str | None = None,
           days: int = 60, project_filter: str | None = None, top: int = 5) -> list[dict]:
    cutoff = (datetime.now() - timedelta(days=days)).timestamp()
    terms = [t for t in re.split(r"\s+", query.strip()) if t]
    candidates = []
    if not PROJECTS.exists():
        return candidates
    for proj in PROJECTS.iterdir():
        if not proj.is_dir():
            continue
        if project_filter and project_filter not in proj.name:
            continue
        memdir = proj / "memory"
        if not memdir.exists():
            continue
        for f in memdir.glob("session_*.md"):
            try:
                mtime = f.stat().st_mtime
                if mtime < cutoff:
                    continue
                text = f.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            fm = parse_fm_simple(text)
            if drift and fm.get("drift") != drift:
                continue
            if type_filter and fm.get("type") != type_filter:
                continue
            sc = score_session(text, fm, terms)
            if sc <= 0 and terms:
                continue
            candidates.append({
                "path": f, "project": proj.name, "mtime": mtime,
                "fm": fm, "score": sc,
                "preview": text[len(fm) * 6 + 30:][:300] if fm else text[:300],
            })
    candidates.sort(key=lambda r: (r["score"], r["mtime"]), reverse=True)
    return candidates[:top]
```

## 逐行讲解 / What's happening

1. **第 30-41 行 / Lines 30-41 (minimal frontmatter)**:
   - 中文: 它没有依赖完整 YAML parser，只解析 `key: value`，换来部署简单和可预测行为。
   - English: It avoids a full YAML dependency and parses only `key: value`, trading syntax coverage for simple, predictable deployment.
2. **第 44-57 行 / Lines 44-57 (weighted score)**:
   - 中文: name 权重 3，description 权重 2，body 权重 1；body 用平方根让重复命中递减。
   - English: Names get weight 3, descriptions weight 2, and body weight 1; the square root makes repeated body hits diminish.
3. **第 62-90 行 / Lines 62-90 (cheap filters first)**:
   - 中文: 先按时间、project、drift、type 过滤，避免把所有文件都送进后续排序。
   - English: Time, project, drift, and type filters run before scoring, keeping irrelevant files out of ranking.
4. **第 91-97 行 / Lines 91-97 (stable top-k)**:
   - 中文: 候选按 `(score, mtime)` 倒序；同分时新文件胜出，结果既相关又新鲜。
   - English: Candidates sort by `(score, mtime)` descending; newer files win ties, keeping results both relevant and fresh.

## 类比 / The analogy

像整理一个工程师的纸质实验记录：项目名写在封面最醒目，摘要写在目录页，正文里的重复词只算作弱证据；先按日期和标签筛掉不相关记录，再把最像的几本放到桌上。

It is like searching an engineer's paper lab notebook. The project name on the cover is strongest, the summary page is next, and repeated words in the body are weak evidence. Filter by date and labels first, then put the closest few notebooks on the desk.

## 自己跑一遍 / Try it yourself

```python
def score(name, desc, body, terms):
    return sum(3 * name.count(t) + 2 * desc.count(t) + body.count(t) ** 0.5 for t in terms)

items = [
    ("robotics", "robotics bugfix", "robotics " * 9),
    ("misc", "small note", "robotics " * 20),
]
print([round(score(*item, ["robotics"]), 2) for item in items])
```

运行 / Run with:

```bash
python try.py
```

预期输出 / Expected output:

```text
[8.0, 4.47]
```

中文: 第一条记录虽然正文重复更少，但标题和摘要命中让它排在前面。 / English: The first record wins despite fewer body repetitions because its title and description carry stronger signal.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **BM25** / **BM25**: 中文: 词频有收益但会饱和，和这里的平方根衰减是相同直觉。 / English: Term frequency helps but saturates, which is the same intuition behind the square-root decay here.
- **email search** / **email search**: 中文: subject、sender、body 往往拥有不同权重，先按 folder/date 过滤。 / English: Email search commonly weights subject, sender, and body differently after folder/date filtering.
- **vector reranking pipelines** / **vector reranking pipelines**: 中文: 轻量 lexical score 可以先缩小候选，再交给昂贵的 embedding 或 cross-encoder。 / English: A cheap lexical score can narrow candidates before an expensive embedding or cross-encoder stage.

## 注意事项 / Caveats / when it breaks

- **不是完整 YAML** / **Not full YAML**: 中文: 多行值、列表、嵌套结构可能被错误解析；frontmatter 复杂后应换 parser。 / English: Multiline values, lists, and nested structures can parse incorrectly; switch to a real parser when frontmatter grows.
- **body 截断是启发式** / **Body slicing is heuristic**: 中文: `text[len(text) // 4:]` 不是真正的 frontmatter 边界，会牺牲正文前段的命中。 / English: `text[len(text) // 4:]` is not a true frontmatter boundary and can discard useful matches near the beginning of the body.
- **mtime 不是业务时间** / **mtime is not semantic time**: 中文: 文件复制或恢复会改变 mtime；更可靠的系统应读取 frontmatter timestamp。 / English: Copying or restoring files changes mtime; a stronger system should rank by a frontmatter timestamp.
- **预览偏移不稳定** / **Preview offset is approximate**: 中文: `len(fm) * 6 + 30` 只是估算，适合 fallback，不适合精确高亮。 / English: `len(fm) * 6 + 30` is only an estimate, fine for a fallback preview but not for precise highlighting.

## 延伸阅读 / Further reading

- [Nautilus Compass session search](https://github.com/chunxiaoxx/nautilus-compass/blob/c7c3126c78c0e92769cf263421b37fe58c3bc5c9/session_search.py#L30-L97)
- [Nautilus Compass README](https://github.com/chunxiaoxx/nautilus-compass/blob/c7c3126c78c0e92769cf263421b37fe58c3bc5c9/README.md)
- [BM25 overview](https://en.wikipedia.org/wiki/Okapi_BM25)
