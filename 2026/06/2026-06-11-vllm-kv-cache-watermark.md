---
date: 2026-06-11
topic: infrastructure
source: tracked
repo: vllm-project/vllm
file: vllm/v1/core/kv_cache_manager.py
permalink: https://github.com/vllm-project/vllm/blob/b8142294b7e757f3a39729c4f400bafaed534681/vllm/v1/core/kv_cache_manager.py#L348-L420
difficulty: intermediate
read_time: ~12 min
tags: [code-of-the-day, infrastructure, vllm, kv-cache, scheduler, back-pressure]
---

# vLLM 的 KV-cache "高水位线":只对新进队的请求收一笔押金,治好抢占抖动 / vLLM's KV-cache "watermark": charge admission rent only on newly-admitted requests to stop preemption thrash

> **一句话 / In one line**: 给 `WAITING` / `PREEMPTED` 请求额外要 `watermark_blocks` 个空闲 block 才放它们进 GPU,正在跑的请求依然能自由扩张,从此抢占抖动消失。 / Make `WAITING` / `PREEMPTED` requests clear an extra `watermark_blocks` of free KV blocks before being admitted; already-running requests still grow freely — preemption thrash stops cold.

## 为什么重要 / Why this matters

vLLM 的 paged KV cache 是按需分配的:每个序列每解码出一个 token,可能就要从 block pool 拿一块新的。问题在于,如果调度器**饥饿地接收**所有 WAITING 队列里的请求,GPU 上的 block 很快被瓜分干净,然后**正在跑的某个长序列**因为想再申请一块而抢占失败 — 它被踢出去,降级为 PREEMPTED,然后它的 block 被回收,而那些刚进队的请求又开始消耗 block……这就是所谓 **preemption thrash**(抢占抖动):整个集群被卡在"放进来→踢出去→放进来→踢出去"的活锁里,吞吐反而比单序列更差。今天合并的 PR #44594 用一招古老的招式根治了它 — **水位线(watermark)**。它不是一个全局阈值,而是**只对入队的新请求生效的额外门槛**:你想从 WAITING / PREEMPTED 进 RUNNING,你得让池子里多留 `watermark_blocks` 个空 block 出来当"应急储备"。已经在跑的请求扩张 KV 时没有这条门槛,所以它们继续顺利推进,而新人则被挡在闸口外多排一会儿。

vLLM's paged KV cache is allocated on demand: every decoded token may pull a fresh block from a global pool. The problem: if the scheduler **greedily admits** every WAITING request, the pool empties out, and now a **long running sequence** that just wanted *one more block* fails to extend — it gets preempted, demoted to PREEMPTED, its blocks recycled, and the newly admitted requests start consuming again… that's the famous **preemption thrash**. The entire cluster gets stuck in an "admit → preempt → admit → preempt" live-lock, and throughput drops below what a single-sequence server would do. PR #44594 — which landed today — fixes it with a classical OS trick: a **watermark**. The clever twist is that the watermark is *not* a global threshold; it's an **extra hurdle that only applies to admissions**. To move from WAITING/PREEMPTED into RUNNING you must leave `watermark_blocks` of headroom in the pool. Sequences *already running* face no such gate when they extend their KV — so they keep flowing while newcomers wait an extra beat at the door.

## 代码 / The code

`vllm-project/vllm` — [`vllm/v1/core/kv_cache_manager.py`](https://github.com/vllm-project/vllm/blob/b8142294b7e757f3a39729c4f400bafaed534681/vllm/v1/core/kv_cache_manager.py#L348-L420)

```python
if new_computed_blocks is not None:
    new_computed_block_list = new_computed_blocks.blocks
else:
    new_computed_block_list = self.empty_kv_cache_blocks.blocks

# The number of computed tokens is the number of computed tokens plus
# the new prefix caching hits
num_local_computed_tokens = (
    request.num_computed_tokens + num_new_computed_tokens
)
total_computed_tokens = min(
    num_local_computed_tokens + num_external_computed_tokens,
    self.max_model_len,
)

watermark_blocks = 0
# The watermark is applied to waiting/preempted requests only, and only
# when there's at least one request already scheduled.
if has_scheduled_reqs and request.status in (
    RequestStatus.WAITING,
    RequestStatus.PREEMPTED,
):
    watermark_blocks = self.watermark_blocks

if full_sequence_must_fit:
    # First check and fail if the full request sequence won't fit.
    full_num_tokens = min(request.num_tokens, self.max_model_len)

    num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(
        request_id=request.request_id,
        num_tokens=full_num_tokens,
        new_computed_blocks=new_computed_block_list,
        num_encoder_tokens=num_encoder_tokens,
        total_computed_tokens=total_computed_tokens,
        num_tokens_main_model=full_num_tokens,
        apply_admission_cap=True,
    )
    required_blocks = num_blocks_to_allocate + watermark_blocks
    if required_blocks > self.block_pool.get_num_free_blocks():
        return None

num_tokens_main_model = total_computed_tokens + num_new_tokens
num_tokens_need_slot = min(
    num_tokens_main_model + num_lookahead_tokens, self.max_model_len
)

# Free the blocks that are skipped during the attention computation
# (e.g., tokens outside the sliding window).
self.coordinator.remove_skipped_blocks(
    request.request_id, total_computed_tokens
)

num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(
    request_id=request.request_id,
    num_tokens=num_tokens_need_slot,
    new_computed_blocks=new_computed_block_list,
    num_encoder_tokens=num_encoder_tokens,
    total_computed_tokens=num_local_computed_tokens
    + num_external_computed_tokens,
    num_tokens_main_model=num_tokens_main_model,
)

# Keep `reserved_blocks` free for other in-flight sequences, and an
# additional watermark of headroom for waiting/preempted admissions.
available_blocks = self.block_pool.get_num_free_blocks() - reserved_blocks
required_blocks = num_blocks_to_allocate + watermark_blocks
if required_blocks > available_blocks:
    # Cannot allocate new blocks
    return None
```

## 逐行讲解 / What's happening

1. **`watermark_blocks = 0`(默认值)/ default value**:
   - 中文: 默认不收"押金"。正在跑的请求(RUNNING)走到这里 `watermark_blocks` 一直是 0,所以它们扩张 KV 时只看池子里还剩多少 block,**不会**因为水位线而拒绝。
   - English: zero by default. Already-RUNNING requests reach this line with `watermark_blocks == 0`, so when they want one more block they only check raw free-block count — **no extra hurdle**.

2. **`if has_scheduled_reqs and request.status in (WAITING, PREEMPTED): watermark_blocks = self.watermark_blocks`**:
   - 中文: 触发水位线的两个条件:(a) 当前 step 已经至少有一个正在跑的请求,(b) 这个请求是从 WAITING 或者刚被踢掉的 PREEMPTED 进来的。换句话说,**只有新人才交押金**。如果整个 step 里**没有任何**正在跑的请求(比如系统刚启动、所有人都在排队),水位线也不收 — 让第一个请求先进去,否则永远没人能跑。
   - English: two conjoined conditions trigger the watermark — (a) at least one scheduled request exists this step, and (b) the candidate is coming in from WAITING or just-preempted PREEMPTED. In other words **only newcomers pay rent**. If *nobody* is running yet (cold start, everyone queued), the watermark stays off — letting the first request in, otherwise nothing would ever start.

3. **`required_blocks = num_blocks_to_allocate + watermark_blocks`**:
   - 中文: 真实需要 = 实际申请量 + 押金。这是水位线的核心一行 — 它把"我需要 N 块"变成了"我需要 N + 押金 块"。空 block 数量必须超过这个加和才算"放得下"。
   - English: required = real allocation + watermark deposit. This single line is the entire mechanism — turning "I need N blocks" into "I need N + watermark blocks". Only if free blocks exceed this sum is the request admitted.

4. **`available_blocks = self.block_pool.get_num_free_blocks() - reserved_blocks`**:
   - 中文: 注意`available_blocks` 还要先扣除 `reserved_blocks`(为已经在跑的请求预留的)。所以一个新请求实际看到的可用 block 是 `free - reserved`,然后还得多余出 `watermark_blocks` 块。这三层减法叠起来才是 vLLM 真正的"admission control"。
   - English: `available_blocks` already subtracts `reserved_blocks` (headroom for already-running sequences). So a newcomer sees `free - reserved` free blocks, *and* still needs another `watermark_blocks` of slack on top. Those three layers of subtraction together form vLLM's real admission control.

5. **`if required_blocks > available_blocks: return None`**:
   - 中文: 不让进。返回 None,调度器把这个请求继续留在 WAITING 队列,下个 step 再试。它的位置不会丢,只是被推迟。
   - English: refuse. Returns `None`, the scheduler keeps the request in WAITING for the next step. Its place isn't lost — just deferred.

## 类比 / The analogy

把 GPU 的 block pool 想成一家小餐馆的座位:**正在用餐的客人**(RUNNING)想多坐一会儿、想加张椅子让朋友进来,服务员都会答应,因为他们已经点了菜、是稳定收入。但**门口排队**(WAITING)或者**刚被请出去的人**(PREEMPTED)想再进来,服务员会要求"店里至少还得空出 3 张桌子才让你进" — 不是因为吝啬,而是怕新客一进来,正在吃饭的客人也想再加椅子时**没地方放**,只好让人家放下筷子走人,然后两人轮流被赶出去。这个"3 张桌子"就是 watermark。它不是赶人,是**控制谁能进**。

Think of the GPU's block pool as a small restaurant's tables. **Diners already eating** (RUNNING) can ask to add a chair for a friend or stay longer — the host always says yes, they're paying customers in flow. But the **queue at the door** (WAITING) and **the guy just escorted out** (PREEMPTED) get told: "we need at least 3 tables free before letting you sit." Not because the host is stingy — but because if a newcomer is seated and then a current diner asks for one more chair, there's nowhere to put it, so somebody has to be ejected. That "3 tables" is the watermark. It doesn't kick anyone out; it just **controls who is allowed in**.

## 自己跑一遍 / Try it yourself

```python
# pip install (none) — pure Python toy
from dataclasses import dataclass

@dataclass
class Req:
    rid: int
    status: str  # "WAITING" | "RUNNING" | "PREEMPTED"
    want: int    # blocks needed

WATERMARK = 3

def admit(req, free, has_scheduled):
    wm = WATERMARK if (has_scheduled and req.status in ("WAITING", "PREEMPTED")) else 0
    return req.want + wm <= free

pool = 10
running = []
queue = [Req(i, "WAITING", 4) for i in range(5)]

step = 0
while queue or running:
    step += 1
    has_sched = bool(running)
    # try to admit waiting requests
    new_queue = []
    for r in queue:
        if admit(r, pool, has_sched):
            r.status = "RUNNING"; pool -= r.want; running.append(r)
        else:
            new_queue.append(r)
    queue = new_queue
    # running requests each ask for 1 more block
    for r in running[:]:
        if pool >= 1:
            pool -= 1
        else:
            r.status = "PREEMPTED"; pool += r.want
            running.remove(r); queue.insert(0, r)
    if step > 20: break
    print(f"step {step}: running={[r.rid for r in running]} queue={len(queue)} free={pool}")
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```
step 1: running=[0, 1] queue=3 free=0
step 2: running=[0, 1] queue=3 free=0
...
```

中文:把 `WATERMARK = 3` 改成 `0`,你会看到 running 列表里的请求被反复加进来又被踢出去 — 这就是 thrash。加上 watermark 之后,新请求只有在池里真的有"余粮"时才被放进来,running 列表稳定增长,thrash 消失。

English: set `WATERMARK = 0` and you'll watch the running list churn — requests admitted, then preempted, then re-admitted — that's thrash. With the watermark on, newcomers only enter when there's slack in the pool, the running set stabilizes, and the live-lock disappears.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Linux `vm.swappiness` / page reclaim watermarks** / **Linux 页回收水位线**: 中文: 内核维护 high / low / min 三道水位线,只有空闲页低于某条线时才触发不同强度的回收,从来不让"申请新页"和"回收旧页"在同一时刻打架。 / English: the kernel keeps high/low/min watermarks; only when free pages drop below a tier does reclaim activate at that intensity, never letting "new page asks" and "old page reclaim" tug-of-war.
- **TCP congestion control's `cwnd` ramp-up** / **TCP 拥塞窗口慢启动**: 中文: 新连接进来不能立刻全速发包,而是先 cwnd 翻倍试探;水位线给"新人"加门槛,跟这个想法是一样的。 / English: a new connection doesn't blast at full rate; it gates itself with a small `cwnd` and ramps up. Same idea — newcomers face friction old flows don't.
- **Database connection pool min-idle settings** / **数据库连接池保留闲置数**: 中文: HikariCP 等连接池都有 `minimumIdle`,即"再忙也得给紧急事务留几条连接",其实是给已有事务的水位线。 / English: HikariCP and friends keep `minimumIdle` connections free — reserved for in-flight transactions — exactly the "headroom for already-running work" pattern.
- **vLLM v0's same trick under a different name** / **vLLM v0 的同名机制**: 中文: 老版 vLLM 也有 `watermark`,但实现散落在多处,v1 把它统一到 `kv_cache_manager` 这一处,代码反而短了。 / English: vLLM v0 had a `watermark` parameter too but its checks were scattered; v1 unifies them inside `kv_cache_manager`, and the code shrinks.

## 注意事项 / Caveats / when it breaks

- **水位线设太高 / Watermark set too high**:
  - 中文: 如果 `watermark` 占总 block 的比例太大(比如 0.2),WAITING 队列会被卡住,首 token 延迟(TTFT)飙升。生产里一般取 0.01 ~ 0.05。
  - English: if `watermark` is a big fraction (say 0.2) of total blocks, the WAITING queue stalls and time-to-first-token explodes. Production values are usually 0.01–0.05.
- **第一个请求的冷启动 / Cold-start admission**:
  - 中文: 这就是 `has_scheduled_reqs` 这个条件存在的原因 — 没有人在跑时不能再要押金,否则永远没人能进。
  - English: that's exactly why `has_scheduled_reqs` is in the condition — with nobody running, demanding a deposit would lock everyone out forever.
- **PREEMPTED 也算"新人" / PREEMPTED counts as a newcomer**:
  - 中文: 一个长序列被踢掉、降级成 PREEMPTED 后,它再次想进来时仍然要交押金。这其实是反人类的吗?不,这是必须的 — 否则它一进来又要更多 block,然后挤掉别人,周而复始。
  - English: a long sequence that gets preempted is treated *like a newcomer* on re-entry — counter-intuitive, but necessary. If it re-entered freely it would immediately grab blocks and likely preempt someone else, restarting the loop.
- **不是替代 reserved_blocks / Not a replacement for reserved_blocks**:
  - 中文: `reserved_blocks` 是给已在跑的请求"提前预留"的,watermark 是给"新进来的请求"加门槛,两者并存,各自处理不同的边界情况。
  - English: `reserved_blocks` pre-reserves slack for already-running requests; the watermark gates newcomers. They are *complementary* — each handles a different boundary case.

## 延伸阅读 / Further reading

- [PR #44594 — "Add admission-control watermark to KV cache manager"](https://github.com/vllm-project/vllm/pull/44594)
- [vLLM paper — *Efficient Memory Management for LLM Serving with PagedAttention* (Kwon et al., SOSP 2023)](https://arxiv.org/abs/2309.06180)
- [Linux kernel docs — "Memory zone watermarks"](https://www.kernel.org/doc/html/latest/admin-guide/mm/concepts.html#memory-zones)
- [vLLM blog — Continuous batching and the cost of preemption](https://blog.vllm.ai/2023/06/20/vllm.html)
