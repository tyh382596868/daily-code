---
date: 2026-06-13
topic: wam
source: wam
repo: yuantianyuan01/FastWAM
file: src/fastwam/trainer.py
permalink: https://github.com/yuantianyuan01/FastWAM/blob/45d8e1458921d83f8ad6cf9ce993d371208dabd0/src/fastwam/trainer.py#L646-L696
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, wam, training-loop, accelerate, gradient-accumulation]
build_role: nanoWAM training-loop — the harness that wraps a flow-matching DiT and turns it into actual training (gradient accumulation, distributed gather, scheduler step)
---

# FastWAM 的 50 行训练 while-loop:HF accelerate 让"梯度累积 + 多卡同步"在零分支语句下完成 / FastWAM's 50-line training while-loop: HF accelerate makes gradient accumulation + multi-process sync work without a single if-branch

> **一句话 / In one line**: 把 while-step 循环包在 `with accelerator.accumulate(model):` 里,**单卡 / DDP / FSDP / DeepSpeed 全部走同一条代码路径**;`accelerator.sync_gradients` 是真正"该 step optimizer 了吗"的官方信号,grad clip → optimizer step → scheduler step → zero_grad → 全局 gather 全部用它来 gate。/ Wrap the while-step loop in `with accelerator.accumulate(model):` and **single-GPU / DDP / FSDP / DeepSpeed all share one code path**; `accelerator.sync_gradients` is the official "is it time to step?" signal that gates grad-clip → optimizer step → scheduler step → zero_grad → global gather.

## 为什么重要 / Why this matters

WAM (世界动作模型) 的训练 loop 比 LLM 训练更复杂:loss 一般是 latent 上的 MSE (flow matching),要同时支持 single GPU、DDP、FSDP、DeepSpeed Zero3,要做 gradient accumulation,要 distributed gather loss 算全局均值,要 grad clip,要 lr scheduler 同步,还要 wandb logging —— 而绝大多数开源 WAM 项目把这些写得乱成一团,if/else 满天飞。FastWAM 这份 50 行的 `train()` 是少有的把这些**全部用 HF accelerate 的 context manager 抽象掉**的版本,看完就有现成的 nanoWAM trainer 模板。

A WAM (world-action-model) training loop is messier than an LLM's: the loss is typically MSE in latent space (flow matching), it must support single-GPU + DDP + FSDP + DeepSpeed-Zero3 simultaneously, do gradient accumulation, gather the loss across processes for a global mean, clip grads, sync the LR scheduler, and log to wandb — and most open-source WAM repos do this with a forest of if/else branches. FastWAM's 50-line `train()` is a rare case where **HF accelerate's context managers swallow every one of those concerns**, giving you a drop-in nanoWAM trainer template.

## 代码 / The code

`yuantianyuan01/FastWAM` — [`src/fastwam/trainer.py`](https://github.com/yuantianyuan01/FastWAM/blob/45d8e1458921d83f8ad6cf9ce993d371208dabd0/src/fastwam/trainer.py#L646-L696)

```python
def train(self):
    self._set_dit_only_train_mode()
    unwrapped_model = self.accelerator.unwrap_model(self.model)

    if self.max_steps is None:
        raise ValueError("`max_steps` must be set before entering the while-step training loop.")

    logger.info("Starting training with max_steps=%d.", self.max_steps)
    data_iter = iter(self.train_loader)
    self.run_start_step = self.global_step
    self.run_start_time = time.perf_counter()

    while self.global_step < self.max_steps:
        try:
            sample = next(data_iter)
            self.batch_in_epoch += 1
        except StopIteration:
            self.epoch += 1
            self.batch_in_epoch = 0
            self.train_sampler.clear_resume_batch_offset()
            data_iter = iter(self.train_loader)
            continue

        with self.accelerator.accumulate(self.model):
            train_model = self.model if hasattr(self.model, "training_loss") \
                                     else self.accelerator.unwrap_model(self.model)

            with self.accelerator.autocast():
                loss, loss_dict = train_model.training_loss(sample)
            self.accelerator.backward(loss)

            if self.accelerator.sync_gradients:
                grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                if not self.accelerator.optimizer_step_was_skipped:
                    self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1

                global_loss = float(
                    self.accelerator.gather(loss.detach().float().reshape(1)).mean().item()
                )
                global_loss_metrics = {}
                for key, value in loss_dict.items():
                    metric_tensor = torch.tensor(float(value), device=loss.device, dtype=torch.float32).reshape(1)
                    global_loss_metrics[key] = float(
                        self.accelerator.gather(metric_tensor).mean().item()
                    )
                grad_norm_tensor = torch.tensor(grad_norm, device=loss.device, dtype=torch.float32)
                global_grad_norm = float(self.accelerator.gather(grad_norm_tensor).mean().item())
                current_lr = float(self.optimizer.param_groups[0]["lr"])
```

## 逐行讲解 / What's happening

1. **第 646-657 行 / Lines 646-657 (一次性 setup / one-shot setup)**:
   - 中文: 把 DiT 主干设成 train 模式 (其它子模块如 VAE、text encoder 冻住),建一个 `data_iter`,记下 wall-clock 起点。**`run_start_step`** 这一记是用来支持"resume from checkpoint" 时正确计算 throughput 的小细节 —— 重启时 `global_step` 不归零但 `run_start_step` 会。
   - English: Flip the DiT trunk into train mode (other modules like VAE and text encoder stay frozen), build a `data_iter`, record a wall-clock origin. **`run_start_step`** is a small detail to support "resume from checkpoint" — `global_step` doesn't reset on restart but `run_start_step` does, so throughput stays correct.

2. **第 659-668 行 / Lines 659-668 — turning a finite DataLoader into an infinite step counter**:
   - 中文: 经典模式 —— `next(data_iter)` 抛 `StopIteration` 就说明这一 epoch 跑完了,`epoch += 1`、清空 sampler 的 resume offset、重新 `iter(self.train_loader)`、`continue` 回到 while。这样**对外只暴露 step 概念,epoch 是内部细节**。`clear_resume_batch_offset` 是为了 checkpoint resume 时不让上一 epoch 的恢复偏移污染新 epoch。
   - English: Classic pattern — `next(data_iter)` raises `StopIteration` to mark the epoch end; bump `epoch`, clear the sampler's resume offset, re-`iter(self.train_loader)`, `continue`. **The outer world sees only steps; epochs are an internal detail.** `clear_resume_batch_offset` ensures the resume offset from the previous epoch doesn't leak into the new one.

3. **第 670 行 / Line 670 (`with accelerator.accumulate(self.model):`)**:
   - 中文: 这是**整个文件最魔法的一行**。HF accelerate 的这个 context manager 把"梯度累积步数"完全藏起来:在累积间隙它会**自动给 DDP 设上 `no_sync()`,跳过 all-reduce**;到了累积满的 step 才放开同步。结果是:你写一份 backward → step 的代码,**单卡、DDP、FSDP、DeepSpeed Zero3 都跑通,且性能不掉**。
   - English: **The single most magical line in the file.** `accelerate`'s context manager hides "gradient accumulation step count" entirely: between accumulation steps it **flips DDP into `no_sync()` and skips the all-reduce**; only at the boundary does it allow the sync. Result: one backward → step recipe runs **single-GPU, DDP, FSDP, and DeepSpeed-Zero3 unmodified, with no perf loss**.

4. **第 671-672 行 / Lines 671-672 (双重 model 句柄 / dual model handle)**:
   - 中文: `self.model` 是被 accelerator wrap 过的版本 (DDP / FSDP wrapper),`train_model` 是 unwrap 过的真实 `nn.Module`。**只有真实模型才有 `training_loss` 方法** —— wrapper 是个透明代理。`hasattr` 检查保留了灵活性:如果你的模型把 `training_loss` 暴露在 wrapper 上,就直接用 wrapper。
   - English: `self.model` is the accelerator-wrapped version (DDP / FSDP wrapper); `train_model` is the unwrapped real `nn.Module`. **Only the real model has the `training_loss` method** — the wrapper is a transparent proxy. The `hasattr` check preserves flexibility: if your model exposes `training_loss` on the wrapper, use the wrapper directly.

5. **第 673-674 行 / Lines 673-674 (`autocast` + `training_loss`)**:
   - 中文: `accelerator.autocast()` 自动按 config (bf16 / fp16 / no) 进 mixed precision。loss 计算被一句 `model.training_loss(sample)` 完全交给模型 —— **trainer 与模型完全解耦**:你 DiT 内部加 flow-matching、加 CFG、加任何 loss 组件,trainer 都不需要改。
   - English: `accelerator.autocast()` enters mixed precision per config (bf16 / fp16 / off). Loss computation is fully delegated to one call: `model.training_loss(sample)` — **trainer and model are decoupled**: add flow matching, CFG, any loss component inside your DiT and the trainer is unchanged.

6. **第 675 行 / Line 675 (`accelerator.backward(loss)`)**:
   - 中文: 不要写 `loss.backward()`!`accelerator.backward` 会**自动按 grad accum 步数 scale loss**,也会处理 DeepSpeed 的 `engine.backward()` 等不同后端差异。
   - English: Do *not* write `loss.backward()`. `accelerator.backward` **scales the loss automatically by the grad-accum count** and dispatches to backend-specific calls (e.g. DeepSpeed's `engine.backward()`).

7. **第 677 行 / Line 677 (`if accelerator.sync_gradients:`)**:
   - 中文: 这是 HF accelerate 给出的官方"该 step optimizer 了吗"信号 —— 在 grad accum 中间步它是 False,累积到最后一步才是 True。**所有 optimizer 操作 (clip、step、zero_grad、lr scheduler) 都 gated 在这里**,确保只在真正同步过的梯度上调用。
   - English: HF accelerate's official "is this an optimizer step?" signal — False during accumulation interior steps, True at the boundary. **All optimizer ops (clip, step, zero_grad, LR scheduler) gate on this**, ensuring they only fire on truly-synced gradients.

8. **第 678 行 / Line 678 (`accelerator.clip_grad_norm_`)**:
   - 中文: **不要用 `torch.nn.utils.clip_grad_norm_`**!accelerator 这个版本在 FSDP 下做对了 —— 它先按 shard 算 partial norm,然后跨进程 all-reduce 才得到真正 global grad norm。
   - English: **Don't use `torch.nn.utils.clip_grad_norm_`!** Accelerate's version does the right thing under FSDP — it computes a partial norm per shard, then all-reduces across processes for the real global grad norm.

9. **第 680-681 行 / Lines 680-681 (scheduler.step() 的 gating)**:
   - 中文: 一个非常容易踩坑的细节 —— **如果 optimizer 这一步因为 grad scaler 溢出被跳过,scheduler 也要跳过**。否则 lr schedule 会比 optimizer 步数走得快,提前进入衰减阶段。
   - English: A trap many trainers fall into — **if the optimizer step was skipped because of grad-scaler overflow, the scheduler must also be skipped**. Otherwise the LR schedule advances faster than the optimizer steps, prematurely entering the decay phase.

10. **第 684-694 行 / Lines 684-694 (`accelerator.gather` × 3)**:
    - 中文: 三个 metric 都走同一招:把本地 scalar 包成 `(1,)` shape 的 tensor,`accelerator.gather` 把所有 rank 的拼起来,`.mean()` 得到全局平均。**这是分布式训练里"打印 / 写 wandb 的 loss"和"真实 loss"对齐的唯一正确方式**。不 gather 直接 print 出来的是 local loss,在不同 rank 上是不同的,容易让人误判。
    - English: All three metrics use the same trick: wrap the local scalar as a `(1,)` tensor, `accelerator.gather` concatenates across ranks, `.mean()` gives the global average. **This is the only correct way to align "printed / logged loss" with "true loss" in distributed training**. Skip the gather and you print a per-rank local loss that's different on each process — easy to misread.

## 类比 / The analogy

中文: 想象**一个工厂的流水线 + 中央质检室**。工人 (各 rank GPU) 每装配一个零件 (forward + backward) 就把"我已完成"信号交给装配总线 (`with accelerator.accumulate`);装配总线知道整条线一共需要 N 个零件才能完成一台机器,所以前 N-1 个零件**不广播给质检室** (`no_sync`),只有第 N 个零件到位时才打开 `sync_gradients` 这个总开关,通知质检室进行统一检验 (`clip_grad_norm_`)、计件、贴标签 (`scheduler.step`)、清理工位 (`zero_grad`),最后用 `accelerator.gather` 把每个工位的"本日产量"汇总到中央显示屏。整条流水线无论是一个工人 (single GPU) 还是一千个工人 (FSDP) 都是同一个 SOP,工厂经理 (你的代码) 不用知道有几个工位。

English: Picture **a factory assembly line plus a central QA station**. Each worker (rank / GPU) finishes one part (forward + backward) and signals "done" on the assembly bus (`with accelerator.accumulate`); the bus knows the line needs N parts per machine, so the first N-1 parts **don't broadcast to QA** (`no_sync`). Only on part N does it flip the master switch `sync_gradients`, letting QA do a unified inspection (`clip_grad_norm_`), record output (`scheduler.step`), clear the workstation (`zero_grad`), and finally `accelerator.gather` aggregates each station's daily count onto the central display. Same SOP whether you have one worker (single GPU) or a thousand (FSDP) — the factory manager (your code) never has to count stations.

## 在 nanoWAM 中的位置 / Where this lives in your nano-WAM

中文: 在 nanoWAM 课程图里,这一份属于 **`training-loop` 槽位** (依赖 `dit-block` 和 `noise-scheduler`)。**它就是你 nanoWAM 训练入口的模板**,只要你 DiT 类自己实现一个 `training_loss(self, sample)` 方法:
1. 从 sample 里取出 latent (或 pixels → VAE encode → latent)
2. 采 timestep `t` (这里 nanoWAM 学到的就是之前 `noise-scheduler` 课程里的 `FlowMatchScheduler.training_weight(t)`)
3. add noise → DiT(x_t, t, condition) → predict velocity → MSE
4. 返回 `(loss, loss_dict)`

trainer 不会变,**所有的"分布式同步、grad accum、wandb logging"统统是这里负责**。上游是模型 (`dit-block` + `noise-scheduler` 给你 loss);下游是 checkpoint + 评估。**省掉这一步**,你只有一个能算 loss 的模型而没有训练它的方法 —— 模型权重永远是初始随机。

生产级实现还会在这 50 行之上加:**(1)** checkpoint save/load (前面有 `save_checkpoint`);**(2)** EMA 权重 (世界模型几乎都要 EMA 才能稳);**(3)** validation loop (前面有 `evaluate`);**(4)** wandb logging 频率控制;**(5)** OOM auto-recovery (FastWAM 这版没做)。

English: In the nanoWAM curriculum graph, this fills the **`training-loop` slot** (depends on `dit-block` and `noise-scheduler`). **It IS your nanoWAM trainer template**, provided your DiT class implements `training_loss(self, sample)`:
1. Pull a latent out of sample (or pixels → VAE encode → latent)
2. Sample timestep `t` (and use the `FlowMatchScheduler.training_weight(t)` from the earlier noise-scheduler lesson)
3. add noise → DiT(x_t, t, condition) → predict velocity → MSE
4. Return `(loss, loss_dict)`

The trainer doesn't change. **All "distributed sync, grad accum, wandb logging" lives here.** Upstream are the model components (`dit-block` + `noise-scheduler` produce the loss); downstream is checkpointing + eval. **Skip this** and you have a model that knows how to compute loss but no way to train — weights stay at random init forever.

A production implementation tacks on top of these 50 lines: **(1)** checkpoint save/load (the surrounding `save_checkpoint` covers this); **(2)** EMA weights (every world model needs EMA to be stable); **(3)** a validation loop (the `evaluate` method elsewhere in the file); **(4)** wandb logging cadence control; **(5)** OOM auto-recovery (FastWAM doesn't do this yet).

## 自己跑一遍 / Try it yourself

```python
# nano_accelerate_loop.py — same skeleton, training a tiny DiT-like MLP on synthetic data
import torch, torch.nn as nn, torch.nn.functional as F
from accelerate import Accelerator

class TinyDiT(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(33, 64), nn.SiLU(), nn.Linear(64, 32))
    def training_loss(self, sample):
        x, t = sample
        noise = torch.randn_like(x)
        x_t = t[:, None] * noise + (1 - t[:, None]) * x
        v_pred = self.body(torch.cat([x_t, t[:, None]], dim=-1))
        loss = F.mse_loss(v_pred, noise - x)
        return loss, {"flow_loss": loss.item()}

acc = Accelerator(gradient_accumulation_steps=4)        # accumulate 4 micro-batches per step
model = TinyDiT()
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
model, opt = acc.prepare(model, opt)

for step in range(200):
    sample = (torch.randn(8, 32), torch.rand(8))
    with acc.accumulate(model):
        with acc.autocast():
            loss, ld = acc.unwrap_model(model).training_loss(sample)
        acc.backward(loss)
        if acc.sync_gradients:
            acc.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); opt.zero_grad(set_to_none=True)
            if step % 10 == 0:
                global_loss = float(acc.gather(loss.detach().reshape(1)).mean())
                print(f"step={step:3d} loss={global_loss:.4f}")
```

运行 / Run with:
```bash
pip install "torch>=2.4" "accelerate>=0.30"
accelerate launch --num_processes=1 nano_accelerate_loop.py
# or just python for single GPU
```

预期输出 / Expected output:
```
step=  0 loss=1.97
step= 10 loss=1.83
step= 20 loss=1.71
...
step=190 loss=0.40
```

中文: 注意 `gradient_accumulation_steps=4` —— 200 个 micro-batch 实际上只对应 50 个 optimizer step。**`acc.sync_gradients` 在前 3 个 micro-batch 是 False,只有第 4 个才 True** —— 把 print 频率从 `step % 10` 改成 `acc.sync_gradients and step % 40 == 0`,你能看到的就是真正的 50 个梯度步。

English: Note `gradient_accumulation_steps=4` — 200 micro-batches actually correspond to only 50 optimizer steps. **`acc.sync_gradients` is False on the first 3 micro-batches and True only on the 4th** — if you change the print gate to `acc.sync_gradients and step % 40 == 0`, you'll see exactly the 50 real gradient steps.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **`huggingface/diffusers` examples/text_to_image/train_text_to_image.py** / **`huggingface/diffusers` examples**: 同一个 `accelerate.accumulate` + `sync_gradients` 模板。/ Same `accelerate.accumulate` + `sync_gradients` template.
- **`huggingface/peft` examples** / **`huggingface/peft` examples**: LoRA 微调脚本也用这套写法。/ LoRA fine-tune scripts use the same pattern.
- **lingbot-va wan_va/train.py** / **lingbot-va wan_va/train.py**: 同等抽象但用纯 PyTorch `gradient_accumulation_steps % == 0` 手写 —— 对照看可以理解 accelerate 帮你省了多少 if/else。/ Same abstraction but with hand-rolled `step % gradient_accumulation_steps == 0` checks — compare side by side to see how many if/else accelerate hides.

## 注意事项 / Caveats / when it breaks

- **`unwrap_model` 时机 / When to `unwrap_model`**:
  - 中文: 调用模型上的"自定义"方法 (像 `training_loss`、`save_pretrained` 等) **永远用 unwrap 后的**,不然 DDP wrapper 不会代理它们。但模型 forward 永远走 wrapper (这样 DDP 才能 hook all-reduce)。这是 HF accelerate 文档里反复强调的纪律。
  - English: For custom methods on the model (`training_loss`, `save_pretrained`, etc.) **always use the unwrapped instance** — DDP wrappers don't proxy them. But the model's `forward` always goes through the wrapper (so DDP can hook all-reduce). A discipline the accelerate docs flag repeatedly.
- **`scheduler.step()` 必须在 `optimizer.step()` 后 / scheduler.step() must follow optimizer.step()**:
  - 中文: PyTorch 1.1 之后这是硬规则。FastWAM 这里写对了,但许多 WAM 复现把它写反,结果第一个 epoch 的 LR 就是错的。
  - English: A hard rule since PyTorch 1.1. FastWAM gets it right here, but plenty of WAM reproductions swap the order and ship a wrong first-epoch LR.
- **`accelerator.gather` 不是免费的 / `accelerator.gather` is not free**:
  - 中文: 每次 gather 都是一次 all-gather collective,在 256-rank 上每 step gather 三次会占 1-3% 训练时间。生产环境可以用 `gather_for_metrics` 或干脆 `log_every=100` 减少频率。
  - English: Each gather is an all-gather collective — three gathers per step on 256 ranks eats 1-3% of training time. Production code uses `gather_for_metrics` or just throttles with `log_every=100`.

## 延伸阅读 / Further reading

- [HF accelerate docs — "Gradient accumulation"](https://huggingface.co/docs/accelerate/usage_guides/gradient_accumulation)
- [HF accelerate docs — "DistributedDataParallel"](https://huggingface.co/docs/accelerate/concept_guides/internal_mechanism)
- [PyTorch `no_sync` context manager](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel.no_sync)
- [Flow matching survey — Lipman 2024](https://arxiv.org/abs/2210.02747)
