---
date: 2026-06-02
topic: diffusion
source: trending
repo: inclusionAI/LLaDA2.0-Uni
file: scripts/t2i_generate.py
permalink: https://github.com/inclusionAI/LLaDA2.0-Uni/blob/3457030a9c737f77f38ad5ff657e7659243d3444/scripts/t2i_generate.py#L34-L75
difficulty: beginner
read_time: ~8 min
tags: [code-of-the-day, diffusion, trending, diffusion-llm, two-phase-generation]
---

# 80 行展示「扩散 LLM 出图」的完整流水线 / 80 lines that demonstrate a full "diffusion-LLM image generation" pipeline

> **一句话 / In one line**: 第一阶段让一个掩码扩散 LLM 用 N 步迭代生成离散 VQ token,第二阶段在 GPU 上卸掉 LLM,加载一个连续流匹配解码器把 token 渲染成像素 —— 整个 LLaDA-2.0-Uni 的推理结构就在这 80 行里。 / Phase 1: a masked-diffusion LLM produces discrete VQ tokens through N mask-fill iterations. Phase 2: unload the LLM from GPU, load a continuous flow-matching decoder, render tokens to pixels — LLaDA-2.0-Uni's entire inference pipeline in 80 lines.

## 为什么重要 / Why this matters

2025-2026 年最有意思的扩散方向之一是「**扩散语言模型**」—— LLaDA、Score-Entropy 系列证明了:LLM 不必是自回归的,可以用「随机 mask 一段 token,网络一次性预测被 mask 的位置」的扩散目标训练,迭代几步就能生成连贯文本。LLaDA-2.0-Uni 把这套思路扩展到了多模态:**同一个网络**既能做图像理解(填 mask 出文字),也能做图像生成(填 mask 出 VQ token)。这个 80 行的 `t2i_generate.py` 是观察这套架构最干净的窗口 —— 没有训练代码、没有大量配置、没有 ComfyUI 适配,只有「LLM 出 token + decoder 出图」的两阶段流水线。

One of the most interesting diffusion directions of 2025-2026 is **diffusion language models** — LLaDA, Score-Entropy and similar work showed LLMs don't have to be autoregressive: train with the objective "randomly mask some tokens, predict all masked positions at once" and a few iterations produce coherent text. LLaDA-2.0-Uni extends this to multimodal: **the same network** both understands images (fills mask → text) and generates images (fills mask → VQ tokens). This 80-line `t2i_generate.py` is the cleanest window into that architecture — no training code, no massive config, no ComfyUI integration, just the "LLM emits tokens + decoder renders pixels" two-phase pipeline.

## 代码 / The code

`inclusionAI/LLaDA2.0-Uni` — [`scripts/t2i_generate.py`](https://github.com/inclusionAI/LLaDA2.0-Uni/blob/3457030a9c737f77f38ad5ff657e7659243d3444/scripts/t2i_generate.py#L34-L75)

```python
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prompts = []
    if args.prompt: prompts = [args.prompt]
    elif args.prompts_file:
        with open(args.prompts_file) as f: prompts = [l.strip() for l in f if l.strip()]
    else: raise ValueError("--prompt or --prompts_file required")
    os.makedirs(args.output_dir, exist_ok=True)

    # Phase 1: generate VQ tokens
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map={"": device}, trust_remote_code=True)
    model = model.to(torch.bfloat16).eval()
    model.tokenizer = tokenizer

    results = []
    for i, prompt in enumerate(prompts):
        print(f"[{i+1}/{len(prompts)}] {prompt[:80]}")
        res = model.generate_image(prompt, image_h=args.image_h, image_w=args.image_w,
                                   steps=args.steps, cfg_scale=args.cfg_scale)
        results.append({"prompt": prompt, **res})

    del model; gc.collect(); torch.cuda.empty_cache()
    print("Model unloaded.\n")

    # Phase 2: decode to images
    for i, res in enumerate(results):
        if args.output and len(prompts) == 1:
            out = args.output
        else:
            safe = res["prompt"][:40].replace(" ", "_").replace("/", "")
            out = os.path.join(args.output_dir, f"{i:04d}_{safe}.png")
        print(f"[{i+1}/{len(results)}] Decoding → {out}")
        img = decode_vq_tokens(res["token_ids"], res["h"], res["w"], args.model_path, device,
                               resolution_multiplier=args.resolution_multiplier, num_steps=args.decoder_steps)
        img.save(out)
```

## 逐行讲解 / What's happening

1. **`AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`**:
   - 中文: 这里加载的是 LLaDA-2.0-Uni 7B 主模型 —— 一个用 HuggingFace `AutoModelForCausalLM` 接口暴露的「**外表是 CausalLM,实际是掩码扩散 LM**」。`trust_remote_code=True` 允许仓库的 `modeling_llada.py` 注入自定义 forward 和 `generate_image` 方法。这是新研究模型「借壳 HF 生态」的常见模式。
   - English: Loads the LLaDA-2.0-Uni 7B main model — exposed via HF's `AutoModelForCausalLM` interface but **outwardly a CausalLM, internally a masked diffusion LM**. `trust_remote_code=True` lets the repo's `modeling_llada.py` inject custom forward and `generate_image` methods. Common pattern for new research models piggy-backing on the HF ecosystem.

2. **`model.tokenizer = tokenizer`**:
   - 中文: 一个小细节但暴露了架构意图:tokenizer 被显式挂到模型对象上,因为 `generate_image` 内部需要在生成过程中往 token 序列里插入 special tokens(image start, image end,这些 token 的 ID 由 tokenizer 决定)。
   - English: A tiny detail that reveals architectural intent — the tokenizer is bolted onto the model object because `generate_image` internally needs to splice special tokens (image start, image end, whose IDs come from the tokenizer) into the token sequence during generation.

3. **`model.generate_image(prompt, image_h, image_w, steps=16, cfg_scale=4.0)`**:
   - 中文: 这是整个 phase 1。注意两个参数:`steps=16` 和 `cfg_scale=4.0`。常规自回归 LM 没有这两个参数 —— `steps` 只在**扩散过程**里才有意义,`cfg_scale` 也只有**条件扩散**才用得上。这两个参数的存在本身就告诉你 LLaDA 不是普通 LM。每一步内部:模型对当前 mask 位置预测概率分布,按 confidence 重新决定哪些位置应该被「定型」、哪些继续保持 mask 进入下一步。16 步后所有 VQ token 都定型。
   - English: This is all of phase 1. Look at two parameters: `steps=16` and `cfg_scale=4.0`. A regular autoregressive LM has neither — `steps` only matters for **diffusion processes**, `cfg_scale` only for **conditional diffusion**. Their presence alone tells you LLaDA isn't a normal LM. Inside each step: the model predicts a probability distribution over mask positions, ranks by confidence, "locks in" the high-confidence ones, keeps the rest masked for the next step. After 16 steps all VQ tokens are committed.

4. **`del model; gc.collect(); torch.cuda.empty_cache()`** — **关键三连 / the critical trio**:
   - 中文: 整段最容易被忽视、其实最体现工程考量的一行。7B 的 LLaDA 主模型 + 1B 级的连续解码器一起塞进单张 GPU 很难(bf16 下要 14GB + 2GB,加上 KV cache 可能爆 24GB 显存)。所以代码刻意分两阶段:phase 1 跑完显式 free 掉 LLM,phase 2 才加载 decoder。这是大模型多组件推理的标准 pattern,但很多业余实现忘了 `gc.collect()`(Python 不会立即释放 PyTorch tensor 引用)。
   - English: The line that looks like cleanup boilerplate but is actually the deepest engineering choice in the script. The 7B LLaDA + ~1B continuous decoder won't fit in a single GPU together (bf16: 14GB + 2GB, plus KV cache, easily 24GB+ peak). The script deliberately splits into two phases: phase 1 finishes, explicitly free the LLM, *then* phase 2 loads the decoder. Standard pattern for multi-component large-model inference, but many hobby implementations forget `gc.collect()` — Python doesn't immediately drop PyTorch tensor references.

5. **`decode_vq_tokens(res["token_ids"], res["h"], res["w"], ..., num_steps=50)`**:
   - 中文: Phase 2。`token_ids` 是 phase 1 输出的离散 VQ codebook 索引(类比:每个 16×16 像素 patch 被压成 1 个数字)。`num_steps=50` 又是一个扩散步数,但这次是**连续**流匹配的 50 步 —— 因为 token-to-pixel 用的是一个 VAE-style flow decoder,不是 LM。这就是「discrete diffusion → continuous diffusion」的两段式生成。
   - English: Phase 2. `token_ids` are the discrete VQ codebook indices from phase 1 (think: each 16×16 pixel patch compressed to one integer). `num_steps=50` is another diffusion step count, but this time **continuous** flow matching 50 steps — because token-to-pixel uses a VAE-style flow decoder, not an LM. This is the "discrete diffusion → continuous diffusion" two-stage generation.

6. **`resolution_multiplier=2`**:
   - 中文: Phase 2 还顺手做了 2× 超分。LLM 出的 token 对应原分辨率 latent,decoder 在解码时同时 upscale —— 单一 decoder 网络两件事一起做,省一个独立 SR 模型。
   - English: Phase 2 also does 2× super-resolution. The LLM outputs tokens at the latent resolution, the decoder upscales while decoding — one network does both jobs, sparing you a separate SR model.

## 类比 / The analogy

想象一家两层楼的料理店。一楼是「字幕组」(扩散 LLM):你点一道菜,字幕组用 16 张草稿纸轮流写、改、补,最后写出一份完整的「食谱编号清单」(VQ token ids,每个数字对应中央厨房菜库里的一道半成品)。然后字幕组下班,清单送上二楼。二楼是「厨房」(流匹配 decoder):厨师按清单从菜库里取出 256 个半成品,然后用 50 道工序(`num_steps=50`)把它们煎炒勾芡组合成一道完整菜品(像素图)。两组人不能同时工作 —— 厨房空间有限(显存),字幕组必须先腾地方(`del model`)。

Picture a two-floor restaurant. The ground floor is the "captioning team" (diffusion LLM): you order a dish, the team uses 16 rounds of drafts to write, revise and patch a complete "recipe number list" (VQ token IDs, each integer indexing a half-prepared component in central inventory). Then the captioning team clocks out and the list goes upstairs. The first floor is the "kitchen" (flow-matching decoder): the chef pulls 256 components from inventory by ID, then runs 50 process steps (`num_steps=50`) — sauté, season, plate — to compose the finished dish (pixel image). The two teams can't work simultaneously — the kitchen is tight on space (GPU memory), the captioning team must move out first (`del model`).

## 自己跑一遍 / Try it yourself

```python
# pip install torch transformers accelerate huggingface_hub
# 不需要真的下载 7B 模型 — 只演示「两阶段 + GPU 显存重置」这个模式
import torch, gc

class FakeBigLM(torch.nn.Module):
    def __init__(self): super().__init__(); self.w = torch.nn.Parameter(torch.randn(8000, 8000))
    def generate_image(self, prompt, steps=4): return {"token_ids": torch.randint(0, 1024, (256,)), "h": 16, "w": 16}

class FakeDecoder(torch.nn.Module):
    def __init__(self): super().__init__(); self.w = torch.nn.Parameter(torch.randn(2000, 2000))
    def decode(self, token_ids, h, w): return torch.randn(1, 3, h*16, w*16)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"start: {torch.cuda.memory_allocated()/1e6:.1f} MB" if device=="cuda" else "(cpu mode)")

# Phase 1
lm = FakeBigLM().to(device); res = lm.generate_image("a cat", steps=4)
print(f"after LM: {torch.cuda.memory_allocated()/1e6:.1f} MB" if device=="cuda" else "")
del lm; gc.collect(); torch.cuda.empty_cache() if device=="cuda" else None
print(f"after free: {torch.cuda.memory_allocated()/1e6:.1f} MB" if device=="cuda" else "")

# Phase 2
dec = FakeDecoder().to(device); img = dec.decode(res["token_ids"], res["h"], res["w"])
print(f"after decoder: {torch.cuda.memory_allocated()/1e6:.1f} MB" if device=="cuda" else "")
print(f"final image shape: {img.shape}")
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
start: 0.0 MB
after LM: ~256 MB        # 8000*8000 bf16 params + grads/buffers
after free: ~0 MB        # the critical drop — if this doesn't go to ~0, your gc broke
after decoder: ~16 MB
final image shape: torch.Size([1, 3, 256, 256])
```

去掉中间那一行 `del lm; gc.collect(); torch.cuda.empty_cache()`,你会看到 phase 2 时显存占用是 phase 1 + phase 2 之和。**这就是 LLaDA 的脚本作者刻意分两阶段的原因 —— 不释放就 OOM。**

Delete that single cleanup line and you'll see phase 2 memory usage = phase 1 + phase 2. **That is precisely why the LLaDA author split the script into two phases — without the free, you OOM.**

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Lumina-mGPT / Janus-Pro** / **Lumina-mGPT / Janus-Pro**: 同样的「LLM 出离散 token,再用独立 decoder 转像素」模式,但 LLM 部分是经典自回归而不是扩散 LM。 / Same "LLM emits discrete tokens, separate decoder renders pixels" pattern, but the LLM is classical autoregressive rather than diffusion.
- **OpenAI 的 Sora 内部架构(根据公开 paper 推测)** / **OpenAI Sora's architecture (inferred from public papers)**: 多组件流水线,patch encoder → DiT → super-res decoder,每个 phase 单独管理显存。 / Multi-component pipeline — patch encoder → DiT → super-res decoder, each phase handles memory separately.
- **diffusers 的 `pipeline.enable_sequential_cpu_offload()`** / **diffusers' `pipeline.enable_sequential_cpu_offload()`**: 把同一思想做成 API:每个子模型用完自动迁回 CPU。LLaDA 的脚本是手写版,你看完这段更容易理解 diffusers offload API 背后到底在做什么。 / Packages the same idea as an API: each sub-model is auto-migrated to CPU after use. LLaDA's script is the hand-written version; reading it makes the diffusers offload API far less mysterious.
- **vLLM / SGLang 的 multi-model serving** / **vLLM / SGLang's multi-model serving**: 推理服务器层面也有「pipeline 内不同模型分阶段调度,共享 GPU」的设计;本质是把单进程的 `del model` 升级成跨进程的调度器。 / At the serving layer, the same "multiple models share one GPU via phased scheduling" idea — essentially the single-process `del model` upgraded to a cross-process scheduler.

## 注意事项 / Caveats / when it breaks

- **`trust_remote_code=True` 的安全风险** / **`trust_remote_code=True` security risk**: 会执行远程 repo 的任意 Python。给生产环境用之前要 fork 一份、读完 `modeling_llada.py` 再加载。 / Executes arbitrary Python from the remote repo. For production, fork and read `modeling_llada.py` before loading.
- **`gc.collect()` 不能保证立即释放** / **`gc.collect()` doesn't guarantee immediate release**: 如果 model 被一个闭包、profiler、TensorBoard logger 或 `safetensors` 文件句柄持有,reference count 不归零,显存不会还。`del; gc.collect(); empty_cache()` 三连只是必要条件,不是充分条件 —— 严格的做法是检查 `torch.cuda.memory_allocated()` 是否真的归零。 / If the model is held by a closure, profiler, TensorBoard logger, or safetensors file handle, refcount won't hit zero and memory stays pinned. The `del; gc.collect(); empty_cache()` trio is necessary but not sufficient — robust code asserts `torch.cuda.memory_allocated()` actually dropped.
- **两阶段共享同一个 random seed** / **Both phases share the same seed**: 脚本只在开头 `torch.manual_seed(args.seed)` 一次,但 phase 2 加载 decoder 会 advance RNG state(权重初始化、噪声采样)。如果你要 phase 1 + phase 2 都精确可复现,得在 phase 2 开头再 seed 一次。 / The script calls `torch.manual_seed(args.seed)` once, but phase 2 advances RNG state (decoder weight init, noise sampling). For exact phase-1 + phase-2 reproducibility, reseed at the start of phase 2.
- **`cfg_scale=4.0` 的代价** / **The cost of `cfg_scale=4.0`**: 和连续扩散一样,CFG 在 LLaDA 内部跑两遍 forward(conditional + unconditional),所以 phase 1 的实际计算是 `steps × 2`。是否值得要看任务 —— 简单 prompt 可以降到 1.5。 / Just like continuous diffusion, CFG runs two forwards per step (conditional + unconditional) inside LLaDA, so phase 1's real compute is `steps × 2`. Whether it's worth it depends on the prompt — simple prompts work fine at 1.5.

## 延伸阅读 / Further reading

- [LLaDA paper (Nie et al. 2025)](https://arxiv.org/abs/2502.09992)
- [LLaDA2.0-Uni README + model weights](https://huggingface.co/inclusionAI/LLaDA-2.0-Uni)
- [Score-Entropy Discrete Diffusion (Lou et al. 2023)](https://arxiv.org/abs/2310.16834)
- [diffusers `enable_sequential_cpu_offload` API 文档](https://huggingface.co/docs/diffusers/optimization/memory)
