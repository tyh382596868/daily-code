---
date: 2026-09-14
topic: robotics
source: trending
repo: Noietch/EVA-CLIENT
file: tools/conversion/native.py
permalink: https://github.com/Noietch/EVA-CLIENT/blob/ced18258151813d50e6264c8a376e386ca85b99e/tools/conversion/native.py#L115-L207
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, robotics, trending, dataset-quality, parquet, atomic-publish]
---

# EVA-CLIENT 数据质量分流：先分 episode，再原子发布 / EVA-CLIENT Quality Split: Partition Episodes, Then Publish Atomically

> **一句话 / In one line**: EVA-CLIENT 按 QC verdict 把 LeRobot v2 数据拆成 accepted/rejected 两套目录，并用 staging 目录避免半成品覆盖结果。 / EVA-CLIENT splits LeRobot v2 data into accepted and rejected directories by QC verdict, using staging directories to avoid publishing partial results.

## 为什么重要 / Why this matters

中文：机器人数据清洗最怕“筛选逻辑”和“文件复制逻辑”纠缠在一起：一旦中途失败，输出目录可能只写了一半，之后很难判断哪些 episode 已经处理。EVA-CLIENT 把质量判断、统计汇总、临时目录和最终发布分开。它不仅返回 accepted/rejected 数量，还保留原始 episode index，让清洗后的数据仍能追溯回来源。

English: Robot-data cleaning becomes dangerous when filtering and file copying are mixed together. A mid-run failure can leave a half-written output whose processed episodes are unclear. EVA-CLIENT separates verdict logic, summary accounting, staging, and final publication. It reports accepted/rejected counts while preserving source episode indices for traceability.

## 代码 / The code

`Noietch/EVA-CLIENT` — [`tools/conversion/native.py`](https://github.com/Noietch/EVA-CLIENT/blob/ced18258151813d50e6264c8a376e386ca85b99e/tools/conversion/native.py#L115-L207)

```python
def is_rejected_episode(row: dict[str, Any]) -> bool:
    qc_verdict = str(row.get("qc_verdict", "")).lower()
    if qc_verdict == "pass":
        return False
    if qc_verdict == "fail":
        return True
    return str(row.get("quality", "green")).lower() == "red"


def split_dataset_by_quality(
    source_dir: Path,
    accepted_dir: Path | None = None,
    rejected_dir: Path | None = None,
    *,
    replace_existing: bool = False,
    progress_callback: Callable[[QualityExportProgress], None] | None = None,
) -> QualitySplitSummary:
    source_dir = Path(source_dir).resolve()
    accepted_dir = Path(
        accepted_dir or source_dir.with_name(source_dir.name + "_accepted")
    ).resolve()
    rejected_dir = Path(
        rejected_dir or source_dir.with_name(source_dir.name + "_rejected")
    ).resolve()
    if not (source_dir / "meta" / "episodes.jsonl").is_file():
        raise FileNotFoundError(f"not a LeRobot dataset: {source_dir}")
    if accepted_dir == rejected_dir or source_dir in {accepted_dir, rejected_dir}:
        raise ValueError("source, accepted, and rejected directories must be distinct")
    info = json.loads((source_dir / "meta" / "info.json").read_text())
    _require_lerobot_v21(info, source_dir)
    for output in (accepted_dir, rejected_dir):
        if output.exists() and not replace_existing:
            raise FileExistsError(f"output directory already exists: {output}")
        if output.exists() and not output.is_dir():
            raise NotADirectoryError(f"output path is not a directory: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)

    rows = _read_jsonl(source_dir / "meta" / "episodes.jsonl")
    if not rows:
        raise ValueError("dataset has no episode metadata")
    indices = [int(row["episode_index"]) for row in rows]
    if len(indices) != len(set(indices)):
        raise ValueError("episode indices must be unique")
    accepted_rows = [row for row in rows if not is_rejected_episode(row)]
    rejected_rows = [row for row in rows if is_rejected_episode(row)]
    if progress_callback is not None:
        progress_callback(QualityExportProgress(0, len(rows), "", None))

    accepted_stage = Path(
        tempfile.mkdtemp(prefix=f".{accepted_dir.name}.", dir=accepted_dir.parent)
    )
    rejected_stage = Path(
        tempfile.mkdtemp(prefix=f".{rejected_dir.name}.", dir=rejected_dir.parent)
    )
    try:
        accepted_frames = _export_subset(
            source_dir,
            accepted_stage,
            accepted_rows,
            subset="accepted",
            episodes_offset=0,
            episodes_total=len(rows),
            progress_callback=progress_callback,
        )
        rejected_frames = _export_subset(
            source_dir,
            rejected_stage,
            rejected_rows,
            subset="rejected",
            episodes_offset=len(accepted_rows),
            episodes_total=len(rows),
            progress_callback=progress_callback,
        )
        publish_output_pair(
            ((accepted_dir, accepted_stage), (rejected_dir, rejected_stage)),
            replace_existing=replace_existing,
        )
    except Exception:
        shutil.rmtree(accepted_stage, ignore_errors=True)
        shutil.rmtree(rejected_stage, ignore_errors=True)
        raise

    return QualitySplitSummary(
        source_dir=str(source_dir),
        accepted_dir=str(accepted_dir),
        rejected_dir=str(rejected_dir),
        source_episodes=len(rows),
        accepted_episodes=len(accepted_rows),
        rejected_episodes=len(rejected_rows),
        accepted_frames=accepted_frames,
        rejected_frames=rejected_frames,
        rejected_source_indices=tuple(int(row["episode_index"]) for row in rejected_rows),
    )
```

## 逐行讲解 / What's happening

1. **第 115-121 行 / Lines 115-121 (`is_rejected_episode`)**:
   - 中文：显式 `qc_verdict` 优先；没有 verdict 时才回退到旧的 `quality == "red"` 规则，兼容新旧元数据。
   - English: An explicit `qc_verdict` wins; only missing verdicts fall back to the older `quality == "red"` rule, preserving metadata compatibility.
2. **第 132-144 行 / Lines 132-144 (path and format checks)**:
   - 中文：所有路径先 resolve，然后检查源数据确实是 LeRobot v2，并阻止 source/accepted/rejected 指向同一个目录。
   - English: Paths are resolved first, the source is checked as a LeRobot v2 dataset, and aliasing source/output directories is rejected.
3. **第 145-150 行 / Lines 145-150 (output policy)**:
   - 中文：默认不覆盖已有结果；`replace_existing` 必须显式打开，避免一次误运行破坏上次清洗结果。
   - English: Existing outputs are protected by default. Overwriting requires an explicit `replace_existing` choice.
4. **第 152-159 行 / Lines 152-159 (metadata partition)**:
   - 中文：读取 episode 行、验证 index 唯一，再用同一个 predicate 生成 accepted 和 rejected 两个列表。
   - English: Episode rows are loaded, indices are checked for uniqueness, and one predicate produces the accepted and rejected partitions.
5. **第 163-187 行 / Lines 163-187 (staging)**:
   - 中文：两个临时目录位于目标目录的同一父文件系统中，之后可以作为一组发布；失败时只删除 stage，不污染正式输出。
   - English: Two temporary directories live beside the final outputs, allowing them to be published as a pair. On failure, only staging is removed.
6. **第 188-195 行 / Lines 188-195 (cleanup on exception)**:
   - 中文：任何一个 subset 导出失败，两个 stage 都清理掉，调用方拿不到“半成功”的假结果。
   - English: If either subset export fails, both stages are removed so callers never mistake a partial result for success.
7. **第 197-207 行 / Lines 197-207 (summary)**:
   - 中文：返回值同时包含 episode 数、frame 数和 rejected source indices；统计信息和可追溯性被保存在同一个不可变 summary 里。
   - English: The result carries episode counts, frame counts, and rejected source indices in one immutable summary, combining accounting with traceability.

## 类比 / The analogy

中文：像机场安检后的行李分拣。行李先按标签进入“可装机”和“需要复检”两条临时传送带，所有箱子都摆好并核对数量后，才一次性把两条传送带接入正式装运区。中途断电时，正式区不会出现半箱货。

English: Think of baggage sorting after airport security. Bags move onto temporary “ready” and “recheck” belts, and only after both belts are complete and counted are they connected to the shipping area. If power fails midway, the official area contains no half-published batch.

## 自己跑一遍 / Try it yourself

```python
def rejected(row):
    verdict = row.get("qc_verdict", "").lower()
    if verdict == "pass":
        return False
    if verdict == "fail":
        return True
    return row.get("quality", "green").lower() == "red"


rows = [
    {"episode_index": 2, "qc_verdict": "pass"},
    {"episode_index": 5, "qc_verdict": "fail"},
    {"episode_index": 8, "quality": "red"},
]
accepted = [r["episode_index"] for r in rows if not rejected(r)]
rejected_ids = [r["episode_index"] for r in rows if rejected(r)]
print(accepted, rejected_ids)
```

运行 / Run with:

```bash
python try.py
```

预期输出 / Expected output:

```text
[2] [5, 8]
```

中文：示例体现了优先级：显式 QC verdict 覆盖旧字段，旧字段只承担兼容回退。

English: The example shows the precedence rule: an explicit QC verdict overrides the legacy field, which is used only as a compatibility fallback.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **LeRobot dataset conversion** / **LeRobot dataset conversion**: format writers consume a stable episode subset and preserve source indices. / Format writers consume a stable episode subset and preserve source indices.
- **Database migrations** / **Database migrations**: write to a shadow table, validate, then swap it into place. / Write to a shadow table, validate, then swap it into place.
- **ML data curation pipelines** / **ML data curation pipelines**: accepted and rejected examples need auditable reasons, not just a filtered count. / Accepted and rejected examples need auditable reasons, not just a filtered count.

## 注意事项 / Caveats / when it breaks

- **predicate 要稳定** / **Keep the predicate stable**: 清洗规则改变后，必须记录版本，否则同一数据集会得到无法比较的结果。 / Record rule versions or repeated runs become incomparable.
- **原始 index 不能重编号丢失** / **Do not lose source indices**: 下游 debug 需要能回到原 episode。 / Downstream debugging needs a path back to the original episode.
- **原子发布依赖同文件系统** / **Atomic publication depends on the filesystem**: 跨磁盘 rename 可能不再具备同样语义。 / Cross-filesystem moves may not preserve the same atomicity.

## 延伸阅读 / Further reading

- [EVA-CLIENT native conversion](https://github.com/Noietch/EVA-CLIENT/blob/ced18258151813d50e6264c8a376e386ca85b99e/tools/conversion/native.py)
- [EVA-CLIENT repository](https://github.com/Noietch/EVA-CLIENT)
