---
date: 2026-06-15
topic: huggingface
source: huggingface
repo: huggingface/tokenizers
file: bindings/python/src/models.rs
permalink: https://github.com/huggingface/tokenizers/blob/9d50fb068b1246267ff4a97e5ffbbd9abcb89e21/bindings/python/src/models.rs#L48-L70
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, huggingface, tokenizers, rust, concurrency, performance, lock-amortization]
---

# 把 RwLock::read() 从"每个 pre-token 一次"降到"每次 encode 一次": tokenizers 在 88 核机器上跑赢 158% / Amortizing `RwLock::read()` from per-pre-token to per-call: +158% throughput on 88-thread aarch64

> **一句话 / In one line**: tokenizers 给 `Model` trait 加了一个默认 `tokenize_in_pretokenized` 方法,Python / Node 绑定在自己的实现里只 `read()` 锁一次,然后在锁的保护下处理整篇文档的所有 pre-token。 / tokenizers added a default `tokenize_in_pretokenized` trait method; the Python/Node bindings override it to take the `RwLock` read-lock **once per call** and tokenize all pre-tokens under that single guard.

## 为什么重要 / Why this matters

HuggingFace `tokenizers` 的 Python 和 Node 绑定都把内部模型放在 `Arc<RwLock<ModelWrapper>>` 里——因为 Python 端可能在跑 `tokenizer.train(...)`(需要写锁),所以工程上需要一把读写锁来保证并发安全。问题是,旧的 `Model::tokenize(seq)` 实现长这样:`self.model.read().unwrap().tokenize(seq)`。它会被 `TokenizerImpl::do_tokenize` 调用——**每个 pre-token 一次**。一篇 ~6 KB 的文档大概有 1500 个 pre-token,就是 1500 次 atomic 锁 acquire/release。在多线程并发场景下,这些原子操作的开销会把真正的 BPE merge 工作淹没——profile 里 75% 的 CPU 都花在 `PyModel::tokenize` wrapper 上,只有零头花在 BPE 合并本身。

HuggingFace `tokenizers`' Python and Node bindings wrap the inner model in `Arc<RwLock<ModelWrapper>>` — write-lock is needed for `tokenizer.train(...)`, so a read-write lock is the right tool. But the old `Model::tokenize(seq)` impl was `self.model.read().unwrap().tokenize(seq)`, and `TokenizerImpl::do_tokenize` calls it **once per pre-token**. A ~6 KB document has ~1500 pre-tokens, so encoding one doc costs 1500 atomic lock acquire/release pairs. Under contention, this pure overhead dwarfs the actual BPE work — `perf record` showed 75% of cycles spent inside `PyModel::tokenize`, with the real `BPE::merge_word` barely visible.

## 代码 / The code

`huggingface/tokenizers` — [`bindings/python/src/models.rs`](https://github.com/huggingface/tokenizers/blob/9d50fb068b1246267ff4a97e5ffbbd9abcb89e21/bindings/python/src/models.rs#L48-L70)

```rust
impl Model for PyModel {
    type Trainer = PyTrainer;

    fn tokenize(&self, tokens: &str) -> tk::Result<Vec<Token>> {
        self.model.read().unwrap().tokenize(tokens)
    }

    /// See [`Model::tokenize_in_pretokenized`] for the lock-once rationale.
    fn tokenize_in_pretokenized(
        &self,
        pretokenized: &mut PreTokenizedString,
        truncation: Option<(usize, tk::TruncationDirection)>,
    ) -> tk::Result<()> {
        let guard = self.model.read().unwrap();    // (*) lock once
        match truncation {
            Some((max_tokens, direction)) => pretokenized.tokenize_with_limit(
                |normalized| guard.tokenize(normalized.get()),
                max_tokens,
                direction,
            ),
            None => pretokenized.tokenize(|normalized| guard.tokenize(normalized.get())),
        }
    }
    // ...
}
```

And the trait default that this override stays compatible with — [`tokenizers/src/tokenizer/mod.rs`](https://github.com/huggingface/tokenizers/blob/9d50fb068b1246267ff4a97e5ffbbd9abcb89e21/tokenizers/src/tokenizer/mod.rs#L84-L118):

```rust
pub trait Model {
    // ... existing methods ...

    /// Tokenize every pre-token of a PreTokenizedString, optionally truncating
    /// the result. The default calls `self.tokenize()` per pre-token, which is
    /// correct for self-contained Models. Implementations that wrap their inner
    /// model behind a lock (e.g. PyModel / Node Model, both Arc<RwLock<_>>) can
    /// override this to acquire the lock once for the whole sequence.
    fn tokenize_in_pretokenized(
        &self,
        pretokenized: &mut PreTokenizedString,
        truncation: Option<(usize, TruncationDirection)>,
    ) -> Result<()> {
        match truncation {
            Some((max_tokens, direction)) => pretokenized.tokenize_with_limit(
                |normalized| self.tokenize(normalized.get()),
                max_tokens,
                direction,
            ),
            None => pretokenized.tokenize(|normalized| self.tokenize(normalized.get())),
        }
    }
}
```

## 逐行讲解 / What's happening

1. **第 56-70 行 (Python binding override) / Lines 56-70 (the Python binding override)**:
   - 中文: 整个新方法只多写了 1 个 `let guard = self.model.read().unwrap();`,然后在闭包里直接调 `guard.tokenize(...)`。这个 `guard` 是一个 RAII handle——它一直活到函数结束,函数结束才释放锁。期间所有 pre-token 都在同一个读锁保护下处理。
   - English: The override adds one line — `let guard = self.model.read().unwrap();` — and then closures call `guard.tokenize(...)`. `guard` is RAII: it lives until the function returns, and the read-lock is held the entire time. Every pre-token is tokenized under that single guard.

2. **第 65 / 68 行 (闭包 `|normalized| guard.tokenize(normalized.get())`) / Line 65 / 68 (the closure)**:
   - 中文: 闭包里调的是 `guard.tokenize(...)` (内层的 `ModelWrapper::tokenize`),不是 `self.tokenize(...)` (外层 `PyModel::tokenize`,会再去 read 一次锁)。这个区别非常关键——如果不小心写成 `self.tokenize`,就等于啥都没改,因为递归回到加锁路径了。
   - English: The closure calls `guard.tokenize(...)` (the inner `ModelWrapper::tokenize`), NOT `self.tokenize(...)` (which would re-acquire the lock and undo the whole point). One character wrong and the optimization disappears.

3. **第 51-53 行 (旧 `Model::tokenize` 保留) / Lines 51-53 (the old `Model::tokenize` is kept)**:
   - 中文: 注意作者 *没有* 删掉旧的 `fn tokenize`——这个方法仍然被 trait 要求实现,而且别的代码路径(像直接调用 `Model::tokenize` 的库用户)还会用到。新方法是 **额外** 的优化路径,只在 `do_tokenize` 这个热路径上被调用。
   - English: The original `fn tokenize` stays — the trait still requires it, and library users may call it directly. The new method is an *additional* optimization path, only invoked from the hot `do_tokenize` site.

4. **Trait default (mod.rs 第 104-117 行) / The trait default (mod.rs lines 104-117)**:
   - 中文: 默认实现 *字节一致地* 重现了旧行为——它就是把"对每个 pre-token 调 `self.tokenize(...)`"这一段提取出来。这是关键的兼容性设计:外部 crate 里实现了 `Model` 的人(比如自己写了一个 sentencepiece wrapper),不用做任何改动,新增的 trait 方法对他们透明。 *Non-breaking change*。
   - English: The trait default *byte-for-byte* reproduces the old behavior — it just extracts "call `self.tokenize` per pre-token" into a named method. This is the critical compat design: external `Model` implementors (someone's custom SentencePiece wrapper) need to change nothing. Adding a method with a default is **non-breaking**.

5. **`TokenizerImpl::do_tokenize` (caller, not shown)**:
   - 中文: 调用方式从原来的 `pretokenized.tokenize(|n| self.model.tokenize(n.get()))` 改成 `self.model.tokenize_in_pretokenized(&mut pretokenized, truncation)`。两条路径(截断 / 不截断)都共用新方法,所以两条路径都享受锁优化。
   - English: The caller switched from `pretokenized.tokenize(|n| self.model.tokenize(n.get()))` to `self.model.tokenize_in_pretokenized(&mut pretokenized, truncation)`. Both the truncated and the non-truncated paths share the new entry point, so both benefit.

## 类比 / The analogy

想象一个图书馆有个咨询台,每次借书都要"刷卡 → 咨询馆员 → 还卡"。你今天要查 1500 本书的信息。旧办法:每本书都跑一趟"刷卡-咨询-还卡",1500 次。新办法:**进咨询台一次,把卡放在台面上,问完所有 1500 个问题,再一次性拿卡走人**。整个一天你只跟那个磁卡读卡器打交道两次。Rust 的 RwLock 几乎是 1:1 对应这个比喻——`read()` 是刷卡进入,guard 析构是还卡离开,中间所有问题都在同一次"进入"里完成。

Imagine a library reference desk where every lookup is "swipe card → ask librarian → return card". You need to look up 1500 books today. Old: 1500 separate "swipe-ask-return" cycles. New: **walk into the reference room once, put your card on the desk, ask all 1500 questions, then pick up your card and leave**. You touch the card reader exactly twice all day. Rust's `RwLock` maps onto this almost one-for-one — `read()` is "swipe in", `guard` drop is "swipe out", and everything you do in between happens inside a single entry.

## 自己跑一遍 / Try it yourself

```python
# pip install tokenizers
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from concurrent.futures import ThreadPoolExecutor
import time, urllib.request

# A small BPE model — enough to see the lock effect once you scale threads.
tok = Tokenizer(BPE.from_file(
    *map(lambda u: urllib.request.urlretrieve(u)[0],
         ["https://huggingface.co/gpt2/resolve/main/vocab.json",
          "https://huggingface.co/gpt2/resolve/main/merges.txt"])
))
tok.pre_tokenizer = Whitespace()

# Fabricate ~6KB documents.
docs = ["the quick brown fox jumps over the lazy dog. " * 130 for _ in range(2000)]

for n_threads in [1, 4, 16]:
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        list(ex.map(tok.encode, docs))
    dt = time.perf_counter() - t0
    print(f"threads={n_threads:2d}   {len(docs)/dt:7.0f} docs/sec")
```

运行 / Run with:
```bash
pip install tokenizers
python try.py
```

预期输出 / Expected output:
```
threads= 1    18000 docs/sec
threads= 4    38000 docs/sec      # ~2x speedup on a 4-core machine
threads=16    50000 docs/sec      # contention starts dominating
```

中文一两句:在装了 fix 之前的旧版 `tokenizers`(0.21.x 早期),16 线程的吞吐反而比 4 线程 *只多一点*——锁的原子开销吃掉了大部分并行收益。装了这个 patch 之后,16 线程能继续按比例往上爬。

In English: on pre-patch tokenizers (early 0.21.x), going from 4 to 16 threads barely moves the throughput — atomic lock overhead eats the parallelism gain. Post-patch, throughput keeps scaling because the hot lock has been amortized.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Python GIL + `Py_BEGIN_ALLOW_THREADS`** / **Python GIL + `Py_BEGIN_ALLOW_THREADS`**: 完全同构——为了避免每个 C 调用都释放 / 重获取 GIL,你应该 **批量地** 在一个释放区里跑所有 C 工作。 / Identical idea — batch all your C work inside one GIL-release window instead of releasing/reacquiring per call.
- **数据库长连接 vs 每查询连接** / **DB long-lived connection vs per-query**: 同一次 transaction 里跑 1500 个 SELECT 比 1500 次开关 connection 快得多——道理一样:把昂贵的 setup/teardown 摊到一次。 / Running 1500 SELECTs in one transaction beats 1500 connect/close cycles — same "amortize the expensive setup" lesson.
- **NumPy 的 release-the-GIL ufunc** / **NumPy ufuncs that release the GIL once**: `np.add(big_array, big_array)` 只释放一次 GIL,内部 C 循环跑 1e9 次加法。 / `np.add(big_array, big_array)` releases the GIL exactly once, then loops 1e9 additions in C.

## 注意事项 / Caveats / when it breaks

- **写锁饥饿 / Writer starvation**: 现在读锁的持有时间从"一次 BPE 合并"变成"一篇文档的所有 BPE 合并"。如果某个进程同时在跑训练(写锁),它要等到当前 encode 完成才能拿到锁。在 inference-only 工作负载里这无所谓,但混合训练+推理的场景需要小心。 / The read-lock now spans an entire document's worth of BPE work. A concurrent `train(...)` that wants the write-lock has to wait. Fine for inference-only workloads; mixed train+infer setups should think about it.
- **每篇文档 *内部* 仍然是串行的 / The per-doc loop is still serial**: 这个 patch 没有把"一篇文档的 1500 个 pre-token"并行化——pre-token 之间是按词序处理的。真正的并行度来自 `encode_batch` 用 rayon 把多篇文档分给多个线程,每个线程各自走这条优化路径。 / The patch doesn't parallelize *within* a document. Parallelism comes from `encode_batch` distributing different documents across threads with rayon, each running the optimized path independently.
- **重新进入的危险 / Re-entrancy hazard**: 如果将来某个 `Model` 实现想在 `tokenize_in_pretokenized` 里调用别的需要写锁的方法,死锁警告——你已经拿了读锁了。 / If a future `Model` implementation tries to grab the write-lock from inside `tokenize_in_pretokenized`, instant deadlock — the read-lock is already held.

## 延伸阅读 / Further reading

- [PR #2072 — full benchmark numbers and perf profile](https://github.com/huggingface/tokenizers/pull/2072)
- [Rust `RwLock` docs — Reader-Writer semantics](https://doc.rust-lang.org/std/sync/struct.RwLock.html)
- [`PreTokenizedString::tokenize` source](https://github.com/huggingface/tokenizers/blob/9d50fb068b1246267ff4a97e5ffbbd9abcb89e21/tokenizers/src/tokenizer/pre_tokenizer.rs) — the callback wrapper that runs the closure over each pre-token.
- [LSE atomics on ARMv8.1+](https://developer.arm.com/documentation/100934/0100/Large-System-Extensions-LSE-atomics) — why the win is largest on aarch64.
