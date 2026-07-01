---
date: 2026-07-01
topic: infrastructure
source: trending
repo: LMCache/LMCache
file: lmcache/utils.py
permalink: https://github.com/LMCache/LMCache/blob/eb29489364817afbae223b3f712b6d47309d11a0/lmcache/utils.py#L398-L560
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, infrastructure, kv-cache, trending]
---

# LMCache 的 CacheEngineKey：KV chunk 的分布式门牌号 / LMCache CacheEngineKey: A Distributed Address for KV Chunks

> **一句话 / In one line**: 一个 KV cache chunk 要能跨 worker、dtype、请求 tag 被稳定寻址，关键就是把这些维度编码进 key。 / A KV-cache chunk needs stable addressing across worker, dtype, and request tags; the key encodes those dimensions.

## 为什么重要 / Why this matters

KV cache 系统最怕“看起来相同其实不同”的块混在一起。LMCache 的 `CacheEngineKey` 把 `model_name`、`world_size`、`worker_id`、`chunk_hash`、`dtype` 和可选 tag 放进 hash/equality/string/dict 序列化路径，保证内存、磁盘和网络消息说的是同一个地址。

KV-cache systems must avoid mixing chunks that look similar but are actually different. LMCache's `CacheEngineKey` puts `model_name`, `world_size`, `worker_id`, `chunk_hash`, `dtype`, and optional tags into hash/equality/string/dict serialization paths so memory, disk, and wire messages agree on the same address.

## 代码 / The code

`LMCache/LMCache` — [`lmcache/utils.py`](https://github.com/LMCache/LMCache/blob/eb29489364817afbae223b3f712b6d47309d11a0/lmcache/utils.py#L398-L560)

```python
@dataclass(slots=True)
class CacheEngineKey:
    model_name: str
    world_size: int
    worker_id: int
    chunk_hash: int
    dtype: torch.dtype
    request_configs: Optional[dict] = field(default_factory=dict)
    tags: Optional[tuple] = field(init=False, default=None)
    _dtype_str: str = field(init=False, default="")

    def __post_init__(self):
        tag_list = None
        if self.request_configs is not None:
            for k, v in self.request_configs.items():
                if k.startswith("lmcache.tag."):
                    if tag_list is None:
                        tag_list = []
                    tag_list.append((k[len("lmcache.tag.") :], v))
        if self.dtype not in TORCH_DTYPE_TO_STR_DTYPE:
            raise ValueError(f"Unsupported dtype in CacheEngineKey: {self.dtype}")
        self._dtype_str = TORCH_DTYPE_TO_STR_DTYPE[self.dtype]
        # use tuple to save tags
        self.tags = None if tag_list is None else tuple(tag_list)

    def __hash__(self):
        return hash(
            (
                self.model_name,
                self.world_size,
                self.worker_id,
                self.chunk_hash,
                self._dtype_str,
                self.tags,
            )
        )

    def __eq__(self, other):
        if type(self) is type(other):
            return (
                self.model_name == other.model_name
                and self.world_size == other.world_size
                and self.worker_id == other.worker_id
                and self.chunk_hash == other.chunk_hash
                and self.dtype == other.dtype
                and self.tags == other.tags
            )

        return False

    def to_string(self):
        s = (
            f"{self.model_name}@{self.world_size}"
            f"@{self.worker_id}@{self.chunk_hash_hex}@{self._dtype_str}"
        )
        if self.tags is not None and len(self.tags) != 0:
            tags = [f"{k}%{v}" for k, v in self.tags]
            s += "@" + "@".join(tags)
        return s

    def split_layers(self, num_layers: int) -> List["LayerCacheEngineKey"]:
        """Split the key into multiple keys for each layer"""
        keys = []
        for layer_id in range(num_layers):
            keys.append(
                LayerCacheEngineKey(
                    model_name=self.model_name,
                    world_size=self.world_size,
                    worker_id=self.worker_id,
                    chunk_hash=self.chunk_hash,
                    dtype=self.dtype,
                    request_configs=self.request_configs,
                    layer_id=layer_id,
                )
            )
        return keys

    def get_first_layer(self) -> "LayerCacheEngineKey":
        """Return the key for the first layer"""
        key = LayerCacheEngineKey(
            model_name=self.model_name,
            world_size=self.world_size,
            worker_id=self.worker_id,
            chunk_hash=self.chunk_hash,
            dtype=self.dtype,
            request_configs=self.request_configs,
            layer_id=0,
        )
        return key

    @staticmethod
    def from_string(s):
        parts = s.split("@")
        if len(parts) < 5:
            raise ValueError(f"Invalid key string: {s}")
        request_configs = None
        if len(parts) >= 6:
            request_configs = {}
            for kv in parts[5:]:
                kvs = kv.split("%", 1)
                if len(kvs) != 2:
                    raise ValueError(f"Invalid key string: {s}")
                request_configs["lmcache.tag." + kvs[0]] = kvs[1]
        return CacheEngineKey(
            model_name=parts[0],
            world_size=int(parts[1]),
            worker_id=int(parts[2]),
            chunk_hash=int(parts[3], 16),
            dtype=STR_DTYPE_TO_TORCH_DTYPE[parts[4]],
            request_configs=request_configs,
        )

    def to_dict(self):
        # Note(Kuntai): this is used for serializing CacheEngineKey via msgpack.
        msg = {
            "__type__": "CacheEngineKey",
            "model_name": self.model_name,
            "world_size": self.world_size,
            "worker_id": self.worker_id,
            "chunk_hash": self.chunk_hash,
            "dtype": self._dtype_str,
        }
        if self.request_configs is not None and len(self.request_configs) != 0:
            msg["request_configs"] = [
                f"{k}%{v}" for k, v in self.request_configs.items()
            ]
        return msg

    @staticmethod
    def from_dict(d):
        request_configs = None
        if request_configs_list := d.get("request_configs"):
            request_configs = {}
            for kv in request_configs_list:
                kvs = kv.split("%", 1)
                if len(kvs) != 2:
                    raise ValueError(f"Invalid key dict: {d}")
                request_configs[kvs[0]] = kvs[1]
        return CacheEngineKey(
            model_name=d["model_name"],
            world_size=d["world_size"],
            worker_id=d["worker_id"],
            chunk_hash=d["chunk_hash"],
            dtype=STR_DTYPE_TO_TORCH_DTYPE[d["dtype"]],
            request_configs=request_configs,
        )

    def with_new_worker_id(self, new_worker_id: int) -> "CacheEngineKey":
        # Reconstruct the cache engine key with new worker id
        return CacheEngineKey(
            self.model_name,
            world_size=self.world_size,
            worker_id=new_worker_id,
            chunk_hash=self.chunk_hash,
            dtype=self.dtype,
            request_configs=self.request_configs,
        )

    @property
    def chunk_hash_hex(self) -> str:
        if isinstance(self.chunk_hash, bytes):
            return self.chunk_hash.hex()
        return f"{self.chunk_hash:x}"
```

## 逐行讲解 / What's happening

1. **第 409-421 行 / Lines 409-421**: 中文: 初始化时从 request config 抽取 `lmcache.tag.*`，并把 dtype 规范化成短字符串。 / English: Initialization extracts `lmcache.tag.*` entries and normalizes dtype into a short string.
2. **第 423-444 行 / Lines 423-444**: 中文: hash 和 equality 使用同一组字段，避免 dict/set 查找和相等判断不一致。 / English: Hash and equality use the same field set, avoiding mismatch between dict/set lookup and equality.
3. **第 448-508 行 / Lines 448-508**: 中文: `to_string` 和 `from_string` 把 key 变成可存储、可传输、可解析的稳定文本。 / English: `to_string` and `from_string` turn the key into stable text that can be stored, transmitted, and parsed.
4. **第 545-560 行 / Lines 545-560**: 中文: `with_new_worker_id` 保留 chunk 身份，只改 worker；`chunk_hash_hex` 统一十六进制展示。 / English: `with_new_worker_id` keeps chunk identity while changing worker; `chunk_hash_hex` standardizes hex display.

## 类比 / The analogy

像快递单号：城市、仓库、货架、包裹哈希和特殊标签都要写清楚，否则同名包裹会送错地方。

It is like a shipping label: city, warehouse, shelf, package hash, and special tags must be explicit or similarly named packages go to the wrong place.


## 自己跑一遍 / Try it yourself

```python
from dataclasses import dataclass
@dataclass(frozen=True)
class Key:
    model: str; worker: int; chunk: int; dtype: str
    def to_string(self):
        return f'{self.model}@{self.worker}@{self.chunk:x}@{self.dtype}'
    @staticmethod
    def from_string(s):
        m,w,c,d=s.split('@'); return Key(m,int(w),int(c,16),d)
k=Key('llama',2,48879,'bf16')
print(k.to_string())
print(Key.from_string(k.to_string()))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
llama@2@beef@bf16
Key(model='llama', worker=2, chunk=48879, dtype='bf16')
```

中文: 这个小例子保留了源码里的关键控制流，但把依赖压到最低，便于你直接观察形状、索引或状态变化。

English: The miniature keeps the original control-flow idea while stripping dependencies down so the shape, index, or state change is visible.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **vLLM block tables** / **vLLM block tables**: 中文: block id 也是 KV cache 的地址层。 / English: Block IDs are also an address layer for KV cache.
- **Content-addressed storage** / **Content-addressed storage**: 中文: Git blob hash 同样用内容身份做稳定引用。 / English: Git blob hashes similarly use content identity for stable references.

## 注意事项 / Caveats / when it breaks

- **tag 顺序 / Tag ordering**: 中文: 如果 request config 的 tag 顺序不稳定，字符串形式也可能不同。 / English: If request-config tag order is unstable, string form can differ.
- **dtype 必须受控 / Dtype must be controlled**: 中文: 不支持的 dtype 会直接报错，这是正确的 fail-fast。 / English: Unsupported dtype raises immediately, which is the right fail-fast behavior.

## 延伸阅读 / Further reading

- Source permalink above.
- Project repository linked from the frontmatter.
