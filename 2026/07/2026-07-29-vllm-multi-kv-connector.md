---
date: 2026-07-29
topic: infrastructure
source: tracked
repo: vllm-project/vllm
file: vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py
permalink: https://github.com/vllm-project/vllm/blob/32a423ac0aad67f94f93e97f73338f484b55faec/vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py#L128-L236
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, infrastructure, kv-cache, serving]
---

# vLLM MultiConnector：多个 KV 后端像一个后端 / vLLM MultiConnector: Make Many KV Backends Look Like One

> **一句话 / In one line**: `MultiConnector` 让 vLLM 可以从第一个命中的 KV connector 加载缓存，同时把新 KV 保存到所有 connector。 / `MultiConnector` lets vLLM load KV cache from the first matching connector while saving new KV to every connector.

## 为什么重要 / Why this matters

LLM serving 里的 KV cache 可能同时存在本机 CPU、远端存储、LMCache 或其他传输后端。调度器不应该知道每种后端的细节；它只需要问“这条 request 有多少 token 可以复用”，然后把 block 分配结果通知出去。

In LLM serving, KV cache may live in local CPU memory, remote storage, LMCache, or another transfer backend. The scheduler should not know every backend detail; it needs one question, "how many tokens can this request reuse?", then it broadcasts the allocation decision.

## 代码 / The code

`vllm-project/vllm` — [`vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py`](https://github.com/vllm-project/vllm/blob/32a423ac0aad67f94f93e97f73338f484b55faec/vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py#L128-L236)

```python
class MultiConnector(KVConnectorBase_V1, SupportsHMA):
    """
    A wrapper for using multiple KVConnectors at the same time.

    The current logic is:
    - Load KV from the first connector that advertises available tokens from
      get_num_new_matched_tokens(), based on the order in the config.
    - Save to all connectors.
    """

    @classmethod
    def requires_piecewise_for_cudagraph(cls, extra_config: dict[str, Any]) -> bool:
        """
        MultiConnector requires PIECEWISE CUDA graph mode if any of its
        child connectors require it.
        """
        connectors_config = extra_config.get("connectors", [])
        for conn_config in connectors_config:
            temp_ktc = KVTransferConfig(**conn_config)
            connector_cls = KVConnectorFactory.get_connector_class(temp_ktc)
            child_extra_config = conn_config.get("kv_connector_extra_config", {})
            if connector_cls.requires_piecewise_for_cudagraph(child_extra_config):
                return True
        return False

    @classmethod
    def all_children_support_hma(cls, kv_transfer_config: "KVTransferConfig") -> bool:
        """Return True only if every configured child connector supports HMA."""
        connectors_config = kv_transfer_config.kv_connector_extra_config.get(
            "connectors", []
        )
        if not connectors_config:
            return False
        for conn_config in connectors_config:
            child_config = KVTransferConfig(
                **{"engine_id": kv_transfer_config.engine_id, **conn_config}
            )
            if not KVConnectorFactory.supports_hma_config(child_config):
                return False
        return True

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(
            vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config
        )

        self._connectors: list[KVConnectorBase_V1] = []
        self._ktc_kv_transfer_config = []
        for connector_cls, temp_config in self._get_connector_classes_and_configs(
            vllm_config
        ):
            self._connectors.append(connector_cls(temp_config, role, kv_cache_config))
            self._ktc_kv_transfer_config.append(temp_config.kv_transfer_config)

        assert vllm_config.kv_transfer_config is not None
        self._all_support_hma = MultiConnector.all_children_support_hma(
            vllm_config.kv_transfer_config
        )
        assert (
            vllm_config.scheduler_config.disable_hybrid_kv_cache_manager
            or self._all_support_hma
        ), "HMA should not be enabled unless all sub-connectors support it"

        self._requests_to_connector: dict[str, int] = {}
        self._extra_async_saves: dict[str, int] = {}
```

## 逐行讲解 / What's happening

1. **第 128-136 行 / Lines 128-136 (`MultiConnector`)**:
   - 中文: 类注释把策略写清楚：加载只选一个命中后端，保存则广播到所有后端。
   - English: The class docstring states the policy: load from one matching backend, save to every backend.
2. **第 138-151 行 / Lines 138-151 (`requires_piecewise_for_cudagraph`)**:
   - 中文: 只要任意子 connector 需要 piecewise CUDA graph，父 connector 就必须提升到同样模式。
   - English: If any child connector requires piecewise CUDA graphs, the wrapper must report the same requirement.
3. **第 153-167 行 / Lines 153-167 (`all_children_support_hma`)**:
   - 中文: HMA 是全员能力；一个子后端不支持，整体就不能开启。
   - English: HMA is an all-or-nothing capability; one unsupported child disables the combined path.
4. **第 179-204 行 / Lines 179-204 (state)**:
   - 中文: `_requests_to_connector` 记录哪条 request 从哪个 connector 加载，`_extra_async_saves` 记录多后端异步保存还差几次完成。
   - English: `_requests_to_connector` records the chosen load backend per request; `_extra_async_saves` tracks remaining async save completions across multiple backends.

## 类比 / The analogy

这像一个快递前台：取件时按柜台顺序问“谁有这个包裹”，第一个有货的柜台交付；寄件时则把副本分发到所有仓库，方便以后就近取。

It is like a parcel desk: for pickup, ask counters in order and use the first one that has the parcel; for drop-off, replicate the parcel to all warehouses so future pickup has more options.

## 自己跑一遍 / Try it yourself

```python
class Connector:
    def __init__(self, name, hits):
        self.name, self.hits, self.saved = name, hits, []
    def matched(self, request):
        return self.hits.get(request, 0)
    def save(self, request):
        self.saved.append(request)

class Multi:
    def __init__(self, connectors):
        self.connectors = connectors
        self.chosen = {}
    def pick_loader(self, request):
        for i, c in enumerate(self.connectors):
            if c.matched(request) > 0:
                self.chosen[request] = i
                return c.name
        return None
    def save_all(self, request):
        for c in self.connectors:
            c.save(request)

m = Multi([Connector("cpu", {}), Connector("remote", {"r1": 8})])
print(m.pick_loader("r1"), m.chosen)
m.save_all("r2")
print([c.saved for c in m.connectors])
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
remote {'r1': 1}
[['r2'], ['r2']]
```

这里最关键的是“load 选一个，save 全广播”。这避免同一条 request 同时从多个后端加载，也保证新缓存会被多个后端看见。

The key behavior is "load from one, save to all." That avoids multiple backends racing to load one request while ensuring fresh cache becomes visible to every backend.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **多级 HTTP cache** / **Tiered HTTP cache**: 浏览器、CDN、源站按顺序命中，写入时可以逐层回填。 / Browsers, CDNs, and origins are checked in order; writes can refill several tiers.
- **数据库读写分离** / **Database read/write splitting**: 读请求可选副本，写请求要传播到复制链路。 / Reads can use a replica, while writes must propagate through replication.

## 注意事项 / Caveats / when it breaks

- **顺序就是策略** / **Order is policy**: connector 配置顺序会决定哪个后端优先加载。 / Connector order decides which backend gets first chance to load.
- **异步完成要计数** / **Async completion needs counting**: 多后端保存时，一个 request 可能要等多个完成信号。 / Multi-backend saves may require several completion events for one request.

## 延伸阅读 / Further reading

- [vLLM KV transfer connectors](https://docs.vllm.ai/)
- [Source permalink](https://github.com/vllm-project/vllm/blob/32a423ac0aad67f94f93e97f73338f484b55faec/vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py#L128-L236)
