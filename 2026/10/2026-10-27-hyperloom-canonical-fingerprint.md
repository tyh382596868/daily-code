---
date: 2026-10-27
topic: infrastructure
source: trending
repo: AMD-AGI/Hyperloom
file: src/hyperloom/inference_optimizer/canonical_fingerprint.py
permalink: https://github.com/AMD-AGI/Hyperloom/blob/a5f6f0dda61507e5749b4f4687085ff945580009/src/hyperloom/inference_optimizer/canonical_fingerprint.py#L31-L81
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, infrastructure, reproducibility]
---

# Hyperloom 规范指纹：等价命令只留一个身份 / Hyperloom Canonical Fingerprint: One Identity for Equivalent Commands

> **一句话 / In one line**: Hyperloom 先规范化 CLI 参数与环境变量，再为实验配置生成稳定短 hash。 / Hyperloom canonicalizes CLI arguments and environment variables before producing a stable short hash.

## 为什么重要 / Why this matters

调优系统若把 `--tp=4` 与 `--tp 4` 当成不同实验，会浪费预算并污染缓存命中。

An optimizer that treats `--tp=4` and `--tp 4` as different experiments wastes budget and breaks cache hits.

## 代码 / The code

`AMD-AGI/Hyperloom` — [`canonical_fingerprint.py`](https://github.com/AMD-AGI/Hyperloom/blob/a5f6f0dda61507e5749b4f4687085ff945580009/src/hyperloom/inference_optimizer/canonical_fingerprint.py#L31-L81)

```python
args_pairs = _args_pairs(str(extra_args or ""))
env_pairs = sorted((str(k), str(v)) for k, v in (extra_envs or {}).items())
payload = json.dumps(payload_obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
return hashlib.sha1(payload.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]
```

## 逐行讲解 / What's happening

参数解析器按 flag 合并并排序；环境变量也排序；紧凑 JSON 使字节表示不受字典插入顺序影响。

The parser merges and sorts flags, sorts environment pairs, and serializes compact JSON so insertion order cannot affect the bytes.

## 类比 / The analogy

像给两张排版不同但内容相同的订单生成同一条物流编号。

It is assigning one shipment ID to two differently formatted but identical orders.

## 自己跑一遍 / Try it yourself

```python
import hashlib, json
cfg = {"tp": "4", "precision": "bf16"}
print(hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8])
```

运行 / Run with `python try.py`。输出 / Expected output: 一个稳定的 8 位 hash。 

## 注意事项 / Caveats / when it breaks

- **哈希不是密钥 / Hashes are not secrets**：短 SHA 适合索引，不适合安全认证。 / A short SHA is an index, not an authentication mechanism.
