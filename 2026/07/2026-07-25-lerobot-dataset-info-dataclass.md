---
date: 2026-07-25
topic: robotics
source: tracked
repo: huggingface/lerobot
file: src/lerobot/datasets/utils.py
permalink: https://github.com/huggingface/lerobot/blob/0d383d09f2051444de211739196a28cc94736861/src/lerobot/datasets/utils.py#L104-L220
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, robotics, dataset-contract]
---

# LeRobot DatasetInfo：把机器人数据集元信息变成有类型契约 / LeRobot DatasetInfo: Turn Robot Dataset Metadata into a Typed Contract

> **一句话 / In one line**: 机器人数据集不是一坨 JSON, 而是带校验、兼容层和序列化规则的契约。 / A robot dataset is not just JSON, but a validated contract with compatibility and serialization rules.

## 为什么重要 / Why this matters

机器人数据集会跨训练脚本、Hub、采集端和回放端流动。`DatasetInfo` 把 `meta/info.json` 的字段集中到一个 dataclass 里, 先校验 `fps` 和 chunk 大小, 再把 JSON 里的 list shape 转成代码里更稳定的 tuple。这样下游策略读取数据时, 不必每个调用点都重新猜字段格式。

Robot datasets move through collectors, training jobs, the Hub, and replay tools. `DatasetInfo` centralizes `meta/info.json` as a dataclass, validates core counters and storage sizes, and converts JSON list shapes into tuple shapes expected by code. Downstream policy code no longer has to rediscover the metadata schema at every call site.

## 代码 / The code

`huggingface/lerobot` -- [`src/lerobot/datasets/utils.py`](https://github.com/huggingface/lerobot/blob/0d383d09f2051444de211739196a28cc94736861/src/lerobot/datasets/utils.py#L104-L220)

```python
@dataclass
class DatasetInfo:
    """Typed representation of the ``meta/info.json`` file for a LeRobot dataset.

    Replaces the previously untyped ``dict`` returned by ``load_info()`` and
    created by ``create_empty_dataset_info()``.  Using a dataclass provides
    explicit field definitions, IDE auto-completion, and validation at
    construction time.
    """

    codebase_version: str
    fps: int
    features: dict[str, dict]

    # Episode / frame counters — start at zero for new datasets
    total_episodes: int = 0
    total_frames: int = 0
    total_tasks: int = 0

    # Storage settings
    chunks_size: int = field(default=DEFAULT_CHUNK_SIZE)
    data_files_size_in_mb: int = field(default=DEFAULT_DATA_FILE_SIZE_IN_MB)
    video_files_size_in_mb: int = field(default=DEFAULT_VIDEO_FILE_SIZE_IN_MB)

    # File path templates
    data_path: str = field(default=DEFAULT_DATA_PATH)
    video_path: str | None = field(default=DEFAULT_VIDEO_PATH)

    # Optional metadata
    robot_type: str | None = None
    splits: dict[str, str] = field(default_factory=dict)
    # OpenAI-style tool schemas declared by the dataset. ``None`` means the
    # dataset doesn't declare any — readers fall back to ``DEFAULT_TOOLS``.
    tools: list[dict] | None = None

    def __post_init__(self) -> None:
        # Coerce feature shapes from list to tuple — JSON deserialisation
        # returns lists, but the rest of the codebase expects tuples.
        for ft in self.features.values():
            if isinstance(ft.get("shape"), list):
                ft["shape"] = tuple(ft["shape"])

        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")
        if self.chunks_size <= 0:
            raise ValueError(f"chunks_size must be positive, got {self.chunks_size}")
        if self.data_files_size_in_mb <= 0:
            raise ValueError(f"data_files_size_in_mb must be positive, got {self.data_files_size_in_mb}")
        if self.video_files_size_in_mb <= 0:
            raise ValueError(f"video_files_size_in_mb must be positive, got {self.video_files_size_in_mb}")

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict.

        Converts tuple shapes back to lists so ``json.dump`` can handle them.
        Drops ``tools`` when unset so existing datasets keep a clean
        ``info.json``.
        """
        d = dataclasses.asdict(self)
        for ft in d["features"].values():
            if isinstance(ft.get("shape"), tuple):
                ft["shape"] = list(ft["shape"])
        if d.get("tools") is None:
            d.pop("tools", None)
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "DatasetInfo":
        """Construct from a raw dict (e.g. loaded directly from JSON).

        Unknown keys are ignored for forward compatibility with datasets that
        carry additional fields (e.g. ``total_videos`` from v2.x). A warning is
        logged when such fields are present.
        """
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(k for k in data if k not in known)
        if unknown:
            logger.warning(f"Unknown fields in DatasetInfo: {unknown}. These will be ignored.")
        return cls(**{k: v for k, v in data.items() if k in known})

    # ---------------------------------------------------------------------------
    # Temporary dict-style compatibility layer
    # Allows existing ``info["key"]`` call-sites to keep working without changes.
    # Once all callers have been migrated to attribute access, remove these.
    # ---------------------------------------------------------------------------
    def __getitem__(self, key: str):
        import warnings

        warnings.warn(
            f"Accessing DatasetInfo with dict-style syntax info['{key}'] is deprecated. "
            f"Use attribute access info.{key} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        try:
            return getattr(self, key)
        except AttributeError as err:
            raise KeyError(key) from err

    def __setitem__(self, key: str, value) -> None:
        import warnings

        warnings.warn(
            f"Setting DatasetInfo with dict-style syntax info['{key}'] = ... is deprecated. "
            f"Use attribute assignment info.{key} = ... instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if not hasattr(self, key):
            raise KeyError(f"DatasetInfo has no field '{key}'")
        setattr(self, key, value)

    def __contains__(self, key: str) -> bool:
        """Check if a field exists (dict-like interface)."""
        return hasattr(self, key)

    def get(self, key: str, default=None):
```

## 逐行讲解 / What's happening

1. **第 104-137 行 / Lines 104-137 (`字段定义`)**:
   - 中文: 把版本、fps、features、计数器、文件模板和可选 tools 声明成显式字段。
   - English: The class declares version, fps, feature schema, counters, path templates, and optional tools as explicit fields.
2. **第 139-153 行 / Lines 139-153 (`__post_init__`)**:
   - 中文: 构造后立即把 feature shape 规整成 tuple, 并拒绝非正数的 fps/chunk size。
   - English: After construction it normalizes feature shapes to tuples and rejects non-positive fps or storage sizes.
3. **第 155-168 行 / Lines 155-168 (`to_dict`)**:
   - 中文: 写回 JSON 时再把 tuple 变回 list, 并在 tools 为空时不污染旧数据集。
   - English: When serializing, tuples become JSON lists again and unset tools are omitted for clean legacy metadata.
4. **第 170-182 行 / Lines 170-182 (`from_dict`)**:
   - 中文: 忽略未知字段但打 warning, 让新版数据集不会直接炸掉旧客户端。
   - English: Unknown fields are ignored with a warning, so newer datasets do not immediately break older clients.
5. **第 189-220 行 / Lines 189-220 (`dict-style shim`)**:
   - 中文: 旧代码还能用 `info["fps"]`, 但会收到 deprecation warning。
   - English: Old `info["fps"]` call sites keep working, but get a deprecation warning.

## 类比 / The analogy

这像给实验室样品贴一张标准标签。标签上有温度、批次、来源和保存方式, 进冰箱前检查一遍, 出库时再转成系统能读的格式。

It is like putting a standard label on every lab sample. The label carries temperature, batch, origin, and storage rules, gets checked before storage, and is converted back into the format the inventory system expects.

## 自己跑一遍 / Try it yourself

```python
from dataclasses import dataclass, asdict

@dataclass
class DatasetInfo:
    fps: int
    features: dict
    tools = None
    def __post_init__(self):
        for ft in self.features.values():
            if isinstance(ft.get("shape"), list):
                ft["shape"] = tuple(ft["shape"])
        if self.fps <= 0:
            raise ValueError("fps must be positive")
    def to_dict(self):
        d = asdict(self)
        for ft in d["features"].values():
            if isinstance(ft.get("shape"), tuple):
                ft["shape"] = list(ft["shape"])
        return d

info = DatasetInfo(30, {"action": {"shape": [7]}})
print(info.features["action"]["shape"], info.to_dict())
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```
(7,) {'fps': 30, 'features': {'action': {'shape': [7]}}}
```

最关键的是内存态和 JSON 态可以不同: 代码里用 tuple, 落盘时用 list。

The important bit is that in-memory and JSON forms can differ: tuples inside Python, lists on disk.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Pydantic/TypedDict 数据入口**: 训练平台常把外部 JSON 先收敛成 typed object, 再进入业务逻辑。 / Training platforms often narrow external JSON into typed objects before business logic sees it.
- **Hub dataset cards**: 公开元数据要能容忍新字段, 但内部读取仍要有稳定字段。 / Public metadata should tolerate new fields while internal readers keep stable fields.

## 注意事项 / Caveats / when it breaks

- **兼容层只是过渡**: 长期保留 `__getitem__` 会让新旧 API 并存太久。 / The dict-style shim is transitional. Keeping it forever prolongs two APIs.
- **校验要足够早**: 如果等训练中才发现 fps 无效, debug 成本会高很多。 / Validation should happen early. Discovering a bad fps inside training is much more expensive.

## 延伸阅读 / Further reading

- LeRobot dataset docs: https://github.com/huggingface/lerobot
- Python dataclasses: https://docs.python.org/3/library/dataclasses.html
