---
date: 2026-06-12
topic: huggingface
source: huggingface
repo: huggingface/datasets
file: src/datasets/packaged_modules/iceberg/iceberg.py
permalink: https://github.com/huggingface/datasets/blob/2c45eab1bb975ac3d846f2aa6217b82adec8eba3/src/datasets/packaged_modules/iceberg/iceberg.py#L20-L141
difficulty: intermediate
read_time: ~11 min
tags: [code-of-the-day, huggingface, datasets, iceberg, pickling, multiprocessing]
---

# HF datasets 接 Apache Iceberg:一场"提取可序列化视图"的精彩外科手术 / HF datasets meets Apache Iceberg: a clean "extract a picklable view" surgical operation

> **一句话 / In one line**: pyiceberg 的 `Catalog` 是个握着 SQLAlchemy 引擎的非 picklable 怪物,新加的 Iceberg builder 用三步把"不可序列化的外部系统"变成"多进程友好的 HF dataset" / pyiceberg's `Catalog` is a non-picklable monster holding live SQLAlchemy engines, and the new Iceberg builder turns it into a multiprocessing-friendly HF dataset in three precise moves.

## 为什么重要 / Why this matters

HF datasets 几乎所有 builder 都假设"输入是磁盘上的文件":parquet, json, csv, webdataset…… 一次性扫一遍 glob 就完事。但 Apache Iceberg 不是文件 —— 是 table format,数据散在云对象存储里、元数据放在 catalog 服务里(Glue / Hive / Nessie / REST),你要"扫"一张 Iceberg 表得先连到 catalog 才能拿到 manifest。这次新合并的 PR #8148 第一次让 HF datasets 真正接入了一个**有状态外部系统**,而且全部 API 都得能撑住 `num_proc > 1` 的多进程分片 —— 也就是 Dill 序列化必须过。怎么把一个握着 DB 连接的 Catalog 变得"可被序列化的视图"?这就是这个文件的全部主题。

Almost every HF datasets builder assumes "input is files on disk": parquet, json, csv, webdataset… one glob scan and you're done. Apache Iceberg isn't files — it's a *table format* where data lives in cloud object storage and metadata lives in a catalog service (Glue / Hive / Nessie / REST). To "scan" an Iceberg table you first connect to the catalog and ask for the manifest. This newly merged PR (#8148) is the first time HF datasets really integrates with a **stateful external system**, and the entire API surface has to survive `num_proc > 1` multiprocessing — which means it must Dill-serialize cleanly. How do you turn a Catalog clutching live DB connections into a "picklable view"? That is the whole subject of this file.

## 代码 / The code

`huggingface/datasets` — [`src/datasets/packaged_modules/iceberg/iceberg.py`](https://github.com/huggingface/datasets/blob/2c45eab1bb975ac3d846f2aa6217b82adec8eba3/src/datasets/packaged_modules/iceberg/iceberg.py#L20-L141)

```python
@dataclass
class IcebergConfig(datasets.BuilderConfig):
    catalog: Optional["Catalog"] = None
    table: Optional[Union[str, Dict[str, str]]] = None
    features: Optional[datasets.Features] = None
    columns: Optional[List[str]] = None
    filters: Optional[Union[str, "BooleanExpression"]] = None
    batch_size: int = 131072
    snapshot_id: Optional[int] = None

    def __post_init__(self):
        super().__post_init__()
        if self.catalog is None:
            raise ValueError("`catalog` must be a pyiceberg Catalog object, but got None.")
        if self.table is None:
            raise ValueError("`table` must be specified, e.g. table='db.my_table'")
        if isinstance(self.table, str):
            self.table = {"train": self.table}
        if self.name == "default":
            catalog_id = f"{self.catalog.__class__.__name__}_{self.catalog.name}"
            table_id = "_".join(sorted(self.table.values()))
            self.name = f"{catalog_id}_{table_id}"

    def create_config_id(self, config_kwargs: dict, custom_features=None) -> str:
        # The catalog object is not picklable (contains SQLAlchemy engines, etc.),
        # so we replace it with a hashable string representation before the
        # parent class hashes config_kwargs via dill.
        config_kwargs = config_kwargs.copy()
        catalog = config_kwargs.pop("catalog", None)
        if catalog is not None:
            config_kwargs["_catalog_id"] = f"{catalog.__class__.__name__}_{catalog.name}"
        filters = config_kwargs.pop("filters", None)
        if filters is not None:
            config_kwargs["_filters_repr"] = repr(filters)
        return super().create_config_id(config_kwargs, custom_features=custom_features)


class Iceberg(datasets.ArrowBasedBuilder, datasets.builder._CountableBuilderMixin):
    BUILDER_CONFIG_CLASS = IcebergConfig

    def _split_generators(self, dl_manager):
        splits = []
        for split_name, table_id in self.config.table.items():
            iceberg_table = self.config.catalog.load_table(table_id)

            scan_kwargs = {}
            if self.config.filters is not None:
                scan_kwargs["row_filter"] = self.config.filters
            if self.config.columns:
                scan_kwargs["selected_fields"] = tuple(self.config.columns)
            if self.config.snapshot_id is not None:
                scan_kwargs["snapshot_id"] = self.config.snapshot_id

            scan = iceberg_table.scan(**scan_kwargs)

            if self.info.features is None:
                arrow_schema = scan.projection().as_arrow()
                self.info.features = datasets.Features.from_arrow_schema(arrow_schema)

            tasks = list(scan.plan_files())

            # Extract picklable scan context for multiprocessing compatibility.
            scan_context = (
                scan.table_metadata,
                scan.io,
                scan.projection(),
                scan.row_filter,
                scan.case_sensitive,
                scan.limit,
            )

            splits.append(
                datasets.SplitGenerator(
                    name=split_name,
                    gen_kwargs={"tasks": tasks, "scan_context": scan_context},
                )
            )

        # Drop the catalog reference so the builder becomes picklable for num_proc > 1.
        self.config.catalog = None
        self.config_kwargs.pop("catalog", None)

        return splits

    def _generate_tables(self, tasks, scan_context):
        from pyiceberg.io.pyarrow import ArrowScan
        table_metadata, io, projected_schema, row_filter, case_sensitive, limit = scan_context
        arrow_scan = ArrowScan(table_metadata, io, projected_schema,
                               row_filter, case_sensitive=case_sensitive, limit=limit)
        for task_idx, task in enumerate(tasks):
            for batch_idx, batch in enumerate(arrow_scan.to_record_batches([task])):
                pa_table = pa.Table.from_batches([batch])
                yield Key(task_idx, batch_idx), self._cast_table(pa_table)
```

## 逐行讲解 / What's happening

整个文件围绕一个核心问题:**pyiceberg 的 `Catalog`、`Expression`、`Scan` 这些核心对象都拿着 live connection / native handle,直接 `pickle.dumps()` 会炸**。HF datasets 想要把数据加载分发到多个进程,就必须让 builder 本身可序列化。代码用三步解开了这个结。

The whole file revolves around one problem: **pyiceberg's core objects — `Catalog`, `Expression`, `Scan` — all hold live connections or native handles, and a direct `pickle.dumps()` blows up**. HF datasets wants to distribute data loading across workers, which means the builder itself must serialize. The code unties this knot in three steps.

1. **第一步:在 `create_config_id` 里替换 catalog / Step 1: replace catalog inside `create_config_id`**:
   - 中文: HF datasets 用 dill 给每个 builder 生成一个 fingerprint(用于缓存命名),fingerprint 是 `config_kwargs` 的 hash。但 `catalog` 进 dill 必炸。第 73-83 行先 `copy()` 一份 `config_kwargs`,然后 `pop("catalog")` 把不可序列化的对象抽出来,换成一个字符串 `_catalog_id = "<ClassName>_<catalog.name>"`。同理对 `filters`(pyiceberg `BooleanExpression`),换成 `repr(filters)`。父类继续按字符串 hash —— fingerprint 仍然是稳定的,只是不依赖 live 对象。
   - English: HF datasets uses Dill to fingerprint every builder (the fingerprint becomes the cache name), and the fingerprint is a hash of `config_kwargs`. But the `catalog` would crash inside Dill. Lines 73-83 first `copy()` `config_kwargs`, then `pop("catalog")` to lift out the unpicklable object and replace it with a string `_catalog_id = "<ClassName>_<catalog.name>"`. Same trick for `filters` (a pyiceberg `BooleanExpression`) → `repr(filters)`. The parent class hashes the strings — fingerprint stays stable, no longer depends on a live object.

2. **第二步:在 `_split_generators` 里抽取 picklable scan context / Step 2: extract a picklable scan_context inside `_split_generators`**:
   - 中文: 主进程一次性做 catalog 调用(`load_table` 和 `scan.plan_files()`),拿到所有需要读的文件列表 `tasks`(每个是个 `FileScanTask`,本身可 pickle)。然后从 `scan` 对象里手动拆出 6 个可序列化的成员到一个 tuple `scan_context`:`table_metadata`, `io`(底层 S3/GCS I/O 句柄,这玩意是 native 但 pyiceberg 重写过 `__reduce__`)、投影 schema、row filter、case_sensitive、limit。`scan` 本身扔掉。
   - English: The main process makes the catalog calls once (`load_table` then `scan.plan_files()`) and gets the list of files to read, `tasks` (each is a `FileScanTask`, which is itself picklable). Then it hand-extracts six picklable members from `scan` into a tuple `scan_context`: `table_metadata`, `io` (a native S3/GCS I/O handle, but pyiceberg has overridden `__reduce__`), projection schema, row_filter, case_sensitive, limit. The `scan` object itself is thrown away.

3. **第三步:`self.config.catalog = None` 这一刀 / Step 3: the `self.config.catalog = None` cut**:
   - 中文: 这是最不起眼但最关键的两行(86-87 行)。`SplitGenerator` 把 `gen_kwargs` 序列化分发给 worker,但 worker 在反序列化时还要重建整个 builder 本身,而 builder 拿着 `self.config`,`self.config` 拿着 `catalog`…… 链式持有不放。所以主进程在生成完 splits 后,**主动**把 `catalog` 设成 `None`、从 `config_kwargs` 里 pop 掉 —— 此后 builder 上下文里再也没有那个握着 DB 连接的引用。
   - English: The most unassuming but most critical two lines (86-87). `SplitGenerator` pickles `gen_kwargs` and ships them to workers, but workers also have to deserialize the builder itself, and the builder holds `self.config`, which holds `catalog`… a chain that won't let go. So the main process, **after** generating the splits, sets `catalog` to `None` and pops it from `config_kwargs`. The builder context no longer holds the DB connection.

4. **`_generate_tables` 在 worker 进程里 / `_generate_tables` runs in the worker process**:
   - 中文: Worker 收到一个 picklable `scan_context` tuple,在本地用 `from pyiceberg.io.pyarrow import ArrowScan` 重新拼出一个 ArrowScan 对象,然后顺序读 `tasks` 列表里的每个文件,产 `pa.Table` —— 注意 worker 这里**完全不碰 catalog**,所以即便集群里 worker 没有 catalog 连接凭证也照样能干活。
   - English: A worker receives a picklable `scan_context` tuple, locally rebuilds an `ArrowScan` via `from pyiceberg.io.pyarrow import ArrowScan`, then sequentially reads each file in `tasks` and yields `pa.Table`. The worker **never touches the catalog** — so even cluster workers without catalog credentials can do the work.

## 类比 / The analogy

想象你是博物馆策展人,接到一个跨国巡展任务:从大英博物馆借展品到 6 个分馆。"大英博物馆"(catalog)是一栋有 24 小时安保的实体建筑 —— 你没法把整栋楼塞进集装箱发出去。但你可以做的是:在伦敦本地造好每一件展品的"完整说明卡"(`table_metadata`、`io`、`row_filter`…),再给每个分馆一份说明卡和对应展品的 IP 地址列表(`tasks`)。分馆只需要拿着说明卡和 IP,就能自己从云端取数据展示 —— 整个过程**没有一个分馆需要直接联系大英博物馆**。第 86-87 行的 `self.config.catalog = None` 就是巡展正式开始前,策展人主动剪断"和大英博物馆的电话线",免得有人误打过去要授权。

Picture yourself as a museum curator running an international tour: artifacts from the British Museum go to 6 satellite venues. The British Museum (`catalog`) is a physical building with 24/7 security — you can't ship the building itself. What you *can* do is, in London, prepare a complete "spec card" for every artifact (`table_metadata`, `io`, `row_filter`, …) and give each satellite venue a copy of the cards plus the IP list of the artifacts they need (`tasks`). The satellite venues fetch from the cloud and put on the show — **none of them ever needs to dial the British Museum directly**. Lines 86-87 (`self.config.catalog = None`) are the curator deliberately cutting the phone line to the museum before the tour begins, so no satellite ever accidentally dials in for authorization.

## 自己跑一遍 / Try it yourself

```python
# Minimal "picklability dance": a fake stateful resource and the extract-the-view trick.
import pickle

class LiveDBConnection:
    """Holds a thread-lock that pickle can't serialize."""
    def __init__(self, name): self.name, self._lock = name, _make_unpicklable()
    def list_files(self): return [f"/data/{i}.parquet" for i in range(3)]

def _make_unpicklable():
    import threading
    return threading.Lock()

class IcebergLikeBuilder:
    def __init__(self, conn): self.conn = conn
    def picklable_view(self):
        # Step 1: snapshot whatever the worker actually needs (filenames + a stable id)
        tasks = self.conn.list_files()
        ctx = {"_conn_id": self.conn.name, "tasks": tasks}
        # Step 2: drop the live handle BEFORE pickling
        self.conn = None
        return ctx

# main process
b = IcebergLikeBuilder(LiveDBConnection("warehouse_prod"))
view = b.picklable_view()
data = pickle.dumps((b, view))                       # both serialize cleanly now
print("pickled bytes:", len(data))
print("recovered:", pickle.loads(data))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```
pickled bytes: ~250
recovered: (<__main__.IcebergLikeBuilder object>, {'_conn_id': 'warehouse_prod', 'tasks': ['/data/0.parquet', ...]})
```

中文:如果把 `self.conn = None` 那行注释掉,`pickle.dumps((b, view))` 直接报 `TypeError: cannot pickle '_thread.lock' object` —— 这正是 PR 里 `self.config.catalog = None` 真正在防的事情。

English: Comment out `self.conn = None` and `pickle.dumps((b, view))` immediately blows up with `TypeError: cannot pickle '_thread.lock' object` — which is precisely what `self.config.catalog = None` in the PR is preventing.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Ray / Dask 的 actor handle / Ray and Dask actor handles**: 远程 actor 的"handle"是 picklable 的轻量代理,而真正的 actor state 留在远端 —— 同样的"在边界处把外部系统替换成 picklable 视图"的思路 / The remote actor "handle" is a picklable lightweight proxy while the real state stays remote — the same "replace the external system with a picklable view at the boundary" idea.
- **PyTorch `DataLoader` 的 `worker_init_fn` / PyTorch `DataLoader`'s `worker_init_fn`**: dataset 对象 picklable 才能交给 num_workers > 0,经典 trap 就是 dataset 持有 SQLite 连接,文档里推荐"在 `worker_init_fn` 里建连接、主对象只存路径" / The dataset must be picklable to support num_workers > 0; the canonical trap is holding a SQLite connection — docs recommend "store only the path on the dataset and open the connection in `worker_init_fn`".
- **Spark Connect / DataFusion 的 LogicalPlan 序列化 / Spark Connect / DataFusion's LogicalPlan serialization**: 把 query plan 序列化成 protobuf,worker 上反序列化后再绑定本地 catalog —— 和 Iceberg builder 这个 `scan_context` 几乎同构 / Serialize the query plan to protobuf, then re-bind it to a local catalog on the worker side — structurally identical to this `scan_context`.
- **JAX 的 `jax.jit` traced function / JAX's `jax.jit` traced function**: closure 里捕获的 unhashable 对象会让 jit 失败,标准 fix 也是"提取 abstract value 而不是 live object" / Capturing an unhashable object in a `jit`'d closure crashes — the standard fix is again "extract the abstract value, not the live object".

## 注意事项 / Caveats / when it breaks

- **`scan.plan_files()` 在主进程跑 → 表大时慢 / `scan.plan_files()` runs in the main process → slow on huge tables**: 整个 manifest 扫描发生在主进程,如果表有上百万 manifest entries,主进程会卡几分钟才把任务分出去。生产部署通常会把 plan 拆 partition 分多次跑 / The whole manifest scan happens in the main process. For tables with millions of manifest entries the main process can stall for minutes before any work goes out. Production typically partitions the plan and runs it in stages.
- **跨进程读不到 catalog → 凭证问题转移 / Workers can't read the catalog → credentials problem shifts**: worker 拿到 `scan.io` 后会自己去 S3/GCS 拿数据,所以**对象存储凭证**还是要在每个 worker 上配好。catalog 凭证省了,object store 凭证省不了 / Workers use `scan.io` to pull from S3/GCS directly, so object-store credentials still need to exist on every worker. Catalog credentials drop out; object-store ones don't.
- **`filters` 是 `BooleanExpression` 时 `repr` 可能不稳定 / When `filters` is a `BooleanExpression`, `repr` may be unstable**: `repr(filters)` 作为 fingerprint 的一部分,如果 pyiceberg 升级让 `__repr__` 改了,缓存会全部失效。用 SQL 字符串风格的 filter 反而更安全 / `repr(filters)` becomes part of the fingerprint, so a pyiceberg upgrade that changes `__repr__` invalidates the whole cache. SQL-string filters are actually safer.

## 延伸阅读 / Further reading

- [PR #8148 — Add Apache Iceberg format support](https://github.com/huggingface/datasets/pull/8148)
- [pyiceberg Catalog protocol](https://py.iceberg.apache.org/api/#catalog)
- [HF datasets Builder protocol](https://huggingface.co/docs/datasets/package_reference/builder_classes)
- [Why pickling works the way it does (RealPython)](https://realpython.com/python-pickle-module/)
