---
date: 2026-07-29
topic: infrastructure
source: trending
repo: tirth8205/code-review-graph
file: code_review_graph/search.py
permalink: https://github.com/tirth8205/code-review-graph/blob/90d760aa23fac0353637d2e8f2a431aa08f14366/code_review_graph/search.py#L308-L466
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, infrastructure, code-search, mcp]
---

# code-review-graph hybrid_search：先排序，再让 agent 读代码 / code-review-graph hybrid_search: Rank First, Let the Agent Read Later

> **一句话 / In one line**: `hybrid_search` 把 FTS、embedding、keyword fallback 和上下文文件 boost 合成一个稳定的代码节点排序。 / `hybrid_search` combines FTS, embeddings, keyword fallback, and context-file boosts into one stable ranking of code nodes.

## 为什么重要 / Why this matters

AI coding agent 最贵的资源不是 grep，而是读错文件后的上下文浪费。一个代码图谱工具如果能先把候选函数排序好，就能让 agent 少读很多无关文件。

For AI coding agents, the expensive part is not grep itself; it is wasting context on the wrong files. If a code-graph tool can rank candidate functions first, the agent reads far less irrelevant code.

## 代码 / The code

`tirth8205/code-review-graph` — [`code_review_graph/search.py`](https://github.com/tirth8205/code-review-graph/blob/90d760aa23fac0353637d2e8f2a431aa08f14366/code_review_graph/search.py#L308-L466)

```python
def hybrid_search(
    store: GraphStore,
    query: str,
    kind: Optional[str] = None,
    limit: int = 20,
    context_files: Optional[list[str]] = None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    _out_mode: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    """Hybrid search combining FTS5 BM25 and vector embeddings via RRF."""
    if not query or not query.strip():
        if _out_mode is not None:
            _out_mode.append("none")
        return []

    conn = store._conn
    fetch_limit = limit * 3

    fts_results: list[tuple[int, float]] = []
    emb_results: list[tuple[int, float]] = []

    try:
        fts_results = _fts_search(conn, query, limit=fetch_limit)
    except Exception as e:
        logger.warning("FTS5 unavailable, will use fallback: %s", e)

    emb_results = _embedding_search(
        store, query, limit=fetch_limit, model=model, provider=provider,
    )

    if fts_results or emb_results:
        lists_to_merge = []
        if fts_results:
            lists_to_merge.append(fts_results)
        if emb_results:
            lists_to_merge.append(emb_results)
        merged = rrf_merge(*lists_to_merge)
        if _out_mode is not None:
            if fts_results and emb_results:
                _out_mode.append("hybrid")
            elif fts_results:
                _out_mode.append("fts")
            else:
                _out_mode.append("semantic")
    else:
        keyword_results = _keyword_search(conn, query, limit=fetch_limit)
        if not keyword_results:
            if _out_mode is not None:
                _out_mode.append("none")
            return []
        if _out_mode is not None:
            _out_mode.append("keyword")
        merged = keyword_results

    kind_boosts = detect_query_kind_boost(query)
    context_set = set(context_files) if context_files else set()

    candidate_ids = [node_id for node_id, _ in merged]
    node_rows: dict[int, Any] = {}
    batch_size = 450
    for i in range(0, len(candidate_ids), batch_size):
        batch = candidate_ids[i:i + batch_size]
        placeholders = ",".join("?" for _ in batch)
        rows = conn.execute(
            f"SELECT * FROM nodes WHERE id IN ({placeholders})",  # nosec B608
            batch,
        ).fetchall()
        for row in rows:
            node_rows[row["id"]] = row
```

## 逐行讲解 / What's happening

1. **第 339-348 行 / Lines 339-348 (input and fan-out)**:
   - 中文: 空 query 直接返回；`fetch_limit = limit * 3` 先多取一些，后面过滤和 boost 才有余地。
   - English: Empty queries return immediately; `fetch_limit = limit * 3` over-fetches so later filters and boosts have room.
2. **第 354-363 行 / Lines 354-363 (FTS + embedding)**:
   - 中文: FTS5 和 embedding 搜索相互独立，任一路失败都不让整体搜索崩掉。
   - English: FTS5 and embedding search are independent; failure in one path does not collapse the whole search.
3. **第 365-389 行 / Lines 365-389 (merge or fallback)**:
   - 中文: 有结果就用 RRF 合并；两个高级路径都没结果时才退回 LIKE keyword 搜索。
   - English: Available ranked lists are merged with RRF; only when both advanced paths return nothing does it fall back to LIKE keyword search.
4. **第 391-407 行 / Lines 391-407 (batch fetch)**:
   - 中文: 先合并 node id，再批量查 `nodes` 表，避免每个候选单独打数据库。
   - English: It merges node IDs first, then batch-fetches node rows instead of querying the database once per candidate.

## 类比 / The analogy

这像招聘筛简历：关键词筛一遍、语义匹配筛一遍、熟人推荐加权，最后 HR 才打开简历细读。

It is like resume screening: keyword search gives one ranking, semantic matching gives another, referrals boost some candidates, and only then does HR read the resumes closely.

## 自己跑一遍 / Try it yourself

```python
def rrf(*ranked_lists, k=60):
    scores = {}
    for items in ranked_lists:
        for rank, item in enumerate(items, 1):
            scores[item] = scores.get(item, 0) + 1 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)

fts = ["parse_file", "parse_bytes", "search_nodes"]
semantic = ["search_nodes", "hybrid_search", "parse_file"]
context_files = {"hybrid_search"}
ranked = rrf(fts, semantic)
boosted = sorted(ranked, key=lambda x: (x in context_files, -ranked.index(x)), reverse=True)
print(ranked)
print(boosted)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
['parse_file', 'search_nodes', 'parse_bytes', 'hybrid_search']
['hybrid_search', 'parse_file', 'search_nodes', 'parse_bytes']
```

这个例子展示了 RRF 和 context boost 的分工：RRF 合并多个排序来源，context boost 把当前任务相关文件再往前推。

The example separates two roles: RRF merges multiple ranking sources, while context boost pushes task-relevant files upward.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **RAG 检索器** / **RAG retrievers**: BM25、dense embedding 和 reranker 经常组合使用。 / BM25, dense embeddings, and rerankers are often combined.
- **IDE symbol search** / **IDE symbol search**: 文件名、符号名和当前打开文件会共同影响排序。 / Filenames, symbol names, and currently open files jointly influence ranking.

## 注意事项 / Caveats / when it breaks

- **fallback 不等于低质量** / **Fallback does not mean useless**: embedding 不可用时，FTS/LIKE 仍能给出可解释结果。 / When embeddings are unavailable, FTS/LIKE can still produce explainable results.
- **boost 要克制** / **Boosts need restraint**: 过强的 context boost 会把真正相关但不在当前文件里的节点压下去。 / Over-strong context boosts can bury relevant nodes outside the current files.

## 延伸阅读 / Further reading

- [code-review-graph repository](https://github.com/tirth8205/code-review-graph)
- [Source permalink](https://github.com/tirth8205/code-review-graph/blob/90d760aa23fac0353637d2e8f2a431aa08f14366/code_review_graph/search.py#L308-L466)
