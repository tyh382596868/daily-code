---
date: 2026-10-09
topic: vla
source: vla
repo: huggingface/lerobot
file: src/lerobot/policies/fineart_vla/modeling_fineart_vla.py
permalink: https://github.com/huggingface/lerobot/blob/b9cb121cb7d4e3e26ec5c906d08dda68d151ba52/src/lerobot/policies/fineart_vla/modeling_fineart_vla.py#L554-L591
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, vla, flex-attention-mask]
build_role: vlm-backbone-wiring advanced variant
---
# FineART-VLA Flex mask：prefix 看历史，动作只看本块 / FineART-VLA Flex Mask: Prefix Sees History, Actions See Their Block

> **一句话 / In one line**: 同一条序列里同时放 VLM prefix 和动作 suffix 时，mask builder 用两套 row 函数分别约束注意力。 / When VLM prefix and action suffix share one sequence, the mask builder uses separate row rules for prefix and action queries.

## 为什么重要 / Why this matters

VLA 的棘手点不只是“把图像、语言、动作拼起来”，而是拼完以后每一类 token 的可见范围都不一样。这个 `_FlexMaskBuilder` 把规则写成两个可编译的 mask 函数：VLM prefix 按有效 token 的累计顺序看历史，动作 token 则看非 FAST prefix 和同一个 action block。

The hard part in a VLA is not merely concatenating image, language, and action tokens. Each token family needs a different visibility rule. `_FlexMaskBuilder` turns those rules into two compiled mask functions: VLM rows attend through valid prefix history, while action rows attend to the non-FAST prefix and their own action block.

## 代码 / The code

`huggingface/lerobot` — [`src/lerobot/policies/fineart_vla/modeling_fineart_vla.py`](https://github.com/huggingface/lerobot/blob/b9cb121cb7d4e3e26ec5c906d08dda68d151ba52/src/lerobot/policies/fineart_vla/modeling_fineart_vla.py#L554-L591)

```python
class _FlexMaskBuilder:
    """Build KI masks while retaining stable compiled mask callables."""

    def __init__(self):
        self._key = None

    def build(self, prefix_pad, prefix_att, non_fast_prefix_len, k, chunk):
        _, create_bm = _get_flex_fns(prefix_pad.device)
        b, p = prefix_pad.shape
        a = k * chunk
        device = prefix_pad.device
        key = (b, p, a, int(non_fast_prefix_len), device)
        if self._key != key:
            self._key = key
            self._pad = torch.empty(b, p, dtype=torch.bool, device=device)
            self._cum = torch.empty(b, p, dtype=torch.long, device=device)
            pad, cum = self._pad, self._cum
            nf = int(non_fast_prefix_len)

            def vlm_rows(bi, h, q_idx, kv_idx):
                kv_p = kv_idx.clamp(max=p - 1)
                ok = (cum[bi, kv_p] <= cum[bi, q_idx]) & pad[bi, kv_p] & pad[bi, q_idx]
                return (kv_idx < p) & ok

            def action_rows(bi, h, q_idx, kv_idx):
                kv_p = kv_idx.clamp(max=p - 1)
                to_prefix = (kv_idx < nf) & pad[bi, kv_p]
                same_block = (q_idx // chunk) == ((kv_idx - p) // chunk)
                return to_prefix | ((kv_idx >= p) & same_block)

            self._vlm_mod, self._action_mod = vlm_rows, action_rows

        self._pad.copy_(prefix_pad)
        self._cum.copy_(torch.cumsum(prefix_att.to(torch.long), dim=1))
        s = p + a
        bm_vlm = create_bm(self._vlm_mod, B=b, H=None, Q_LEN=p, KV_LEN=s, device=device)
        bm_action = create_bm(self._action_mod, B=b, H=None, Q_LEN=a, KV_LEN=s, device=device)
        return bm_vlm, bm_action
```

## 逐行讲解 / What's happening

1. **第 560-570 行 / Lines 560-570 (cache key)**:
   - 中文: 只有 batch、prefix 长度、动作长度等结构变了，才重建内部 buffer 和闭包。
   - English: The builder recreates buffers and closures only when structural dimensions change.
2. **第 573-576 行 / Lines 573-576 (`vlm_rows`)**:
   - 中文: prefix token 只能看累计顺序不超过自己的有效 prefix token。
   - English: Prefix rows can attend only to valid prefix keys whose cumulative order is not ahead of the query.
3. **第 578-582 行 / Lines 578-582 (`action_rows`)**:
   - 中文: 动作 token 可以看非 FAST prefix，也可以看同一个动作块里的 token。
   - English: Action rows can see the non-FAST prefix and tokens in the same action block.

## 类比 / The analogy

像排练一场接力赛。教练区可以看已经发生的交接，运动员只能听教练指令和自己这一棒的队友，不能偷看下一棒怎么跑。

It is like rehearsing a relay race. Coaches can inspect completed handoffs, while runners only hear coach instructions and their own leg of the race; they cannot peek at future legs.

## 在 nanoVLA / nanoWAM 中的位置 / Where this lives in your nano-VLA

中文: 这是 `vlm-backbone-wiring` 的高级变体：prefix 里有图像/语言/状态 token，suffix 里有动作 chunk token，mask 决定谁能看谁。在 nanoVLA 里，它应该放在“把 observation prefix 和 action suffix 接入同一个 transformer”的边界上。没有这层 mask，动作 token 可能偷看未来动作，或者语言 prefix 可能因为 padding 混入错误上下文。

English: This is an advanced `vlm-backbone-wiring` component. The prefix holds image/language/state tokens, the suffix holds action-chunk tokens, and the mask decides which rows can attend to which keys. In a nanoVLA, this belongs at the boundary where observation prefix and action suffix enter the same transformer. Without it, action tokens may leak future action information or padded prefix tokens may contaminate context.

## 自己跑一遍 / Try it yourself

```python
def action_visible(q_idx, kv_idx, prefix_len, non_fast_prefix_len, chunk):
    to_prefix = kv_idx < non_fast_prefix_len
    same_block = (q_idx // chunk) == ((kv_idx - prefix_len) // chunk)
    return to_prefix or (kv_idx >= prefix_len and same_block)

prefix_len, chunk = 5, 3
for q in range(6):
    visible = [k for k in range(prefix_len + 6) if action_visible(q, k, prefix_len, 2, chunk)]
    print(q, visible)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```
0 [0, 1, 5, 6, 7]
1 [0, 1, 5, 6, 7]
2 [0, 1, 5, 6, 7]
3 [0, 1, 8, 9, 10]
4 [0, 1, 8, 9, 10]
5 [0, 1, 8, 9, 10]
```

前三个 action query 属于同一块，所以看到同一组三个 action key。

The first three action queries belong to the same block, so they see the same three action keys.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **openpi / FAST action tokens** / **openpi / FAST action tokens**: prefix/action mask 决定哪些 token 参与 loss、哪些 token 只做条件。 / Prefix/action masks decide which tokens are supervised and which only condition the model.
- **FlexAttention block masks** / **FlexAttention block masks**: 把 Python 规则编译成 block mask，能避免每步重建大矩阵。 / Python visibility rules can be compiled into block masks instead of rebuilding dense masks.

## 注意事项 / Caveats / when it breaks

- **闭包依赖 buffer** / **Closures depend on buffers**: `_pad` 和 `_cum` 被原地更新，规则函数读取的是最新 buffer。 / `_pad` and `_cum` are updated in place, and the rule closures read the latest buffers.
- **结构变化才安全复用** / **Reuse only across same structure**: key 里包含 batch、长度和 device，漏掉结构维度会复用错误 mask。 / The cache key must include structural dimensions, or a stale mask can be reused incorrectly.

## 延伸阅读 / Further reading

- [LeRobot FineART-VLA mask builder](https://github.com/huggingface/lerobot/blob/b9cb121cb7d4e3e26ec5c906d08dda68d151ba52/src/lerobot/policies/fineart_vla/modeling_fineart_vla.py#L554-L591)
- [PyTorch FlexAttention](https://pytorch.org/docs/stable/nn.attention.flex_attention.html)
