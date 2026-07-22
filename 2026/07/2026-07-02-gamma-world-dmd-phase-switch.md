---
date: 2026-07-02
topic: robotics
source: trending
repo: nv-tlabs/Gamma-World
file: gamma_world/_src/gamma_world/self_forcing/dmd.py
permalink: https://github.com/nv-tlabs/Gamma-World/blob/6a95de85c439d8ea73eae34c88fbfd4e89ea02e2/gamma_world/_src/gamma_world/self_forcing/dmd.py#L333-L389
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, robotics, world-model, dmd]
---

# Gamma-World DMD 训练相位：学生和 fake score 轮流更新 / Gamma-World DMD Training Phases: Alternate Student and Fake Score Updates

> **一句话 / In one line**: 一个 `is_student_phase` 谓词同时控制 EMA、optimizer、scheduler 和 zero-grad，让 DMD 训练按固定节奏切换目标。 / One `is_student_phase` predicate controls EMA, optimizer, scheduler, and zero-grad so DMD training switches objectives on a fixed rhythm.

## 为什么重要 / Why this matters

自蒸馏和 DMD 训练常有多个网络：生成器、真实 score、fake score、EMA。最容易出错的是“今天该更新谁”。Gamma-World 把相位判断集中到一个函数，后续所有训练钩子都问同一个问题，降低了 optimizer/scheduler 不一致的风险。

Self-distillation and DMD training often involve multiple networks: generator, real score, fake score, and EMA. A common failure mode is updating the wrong component. Gamma-World centralizes the phase decision in one function, then every training hook asks the same question, reducing optimizer/scheduler mismatch risk.

## 代码 / The code

`nv-tlabs/Gamma-World` — [`dmd.py`](https://github.com/nv-tlabs/Gamma-World/blob/6a95de85c439d8ea73eae34c88fbfd4e89ea02e2/gamma_world/_src/gamma_world/self_forcing/dmd.py#L333-L389)

```python
    def is_student_phase(self, iteration: int):
        return (
            self.config.dfake_warm_up_steps == -1 or iteration > self.config.dfake_warm_up_steps
        ) and iteration % self.config.dfake_gen_update_ratio == 0

    def on_before_zero_grad(
        self, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler, iteration: int
    ) -> None:
        if not self.is_student_phase(iteration):
            return

        if self.config.ema.enabled:
            ema_beta = self.ema_beta(iteration)
            self.net_ema_worker.update_average(self.net, self.net_ema, beta=ema_beta)

    def ema_beta(self, iteration: int) -> float:
        if iteration < self.config.ema_start_step:
            return 0.0
        return self.config.ema_weight

    def get_optimizers(self, iteration: int) -> list[torch.optim.Optimizer]:
        if self.is_student_phase(iteration):
            return [self.optimizer_dict["net"]]
        else:
            return [self.optimizer_dict["fake_score"]]

    def get_lr_schedulers(self, iteration: int) -> list[torch.optim.lr_scheduler.LRScheduler]:
        if self.is_student_phase(iteration):
            return [self.scheduler_dict["net"]]
        else:
            return [self.scheduler_dict["fake_score"]]

    def optimizers_schedulers_step(self, grad_scaler: torch.cuda.amp.GradScaler, iteration: int) -> None:
        for optimizer in self.get_optimizers(iteration):
            optimizer.step()

        for scheduler in self.get_lr_schedulers(iteration):
            scheduler.step()

    def optimizers_zero_grad(self, iteration: int) -> None:
        for optimizer in self.get_optimizers(iteration):
            optimizer.zero_grad()
```

## 逐行讲解 / What's happening

1. **第 333-337 行 / Lines 333-337**: 中文: warmup 后，每隔 `dfake_gen_update_ratio` 次迭代进入 student/generator phase。 / English: After warmup, every `dfake_gen_update_ratio` iteration enters the student/generator phase.
2. **第 343-350 行 / Lines 343-350**: 中文: 只有 student phase 才更新 EMA，fake-score phase 不碰 teacher。 / English: EMA updates only during the student phase; fake-score phases leave the teacher alone.
3. **第 358-371 行 / Lines 358-371**: 中文: optimizer 和 scheduler 都从同一相位函数派生，避免“更新 A 但 step B scheduler”。 / English: Optimizers and schedulers derive from the same phase predicate, avoiding "update A but step B's scheduler."
4. **第 373-389 行 / Lines 373-389**: 中文: step 和 zero_grad 都通过 selector 取当前 active optimizer。 / English: Both stepping and zeroing go through the selector for the active optimizer.

## 类比 / The analogy

像健身房的轮换训练表：周一练腿就只调腿部器械和计时器，周二练背就换另一套。所有教练都看同一张表，才不会有人把腿部训练的计时器用到背部训练。

It is like a gym rotation schedule: leg day uses leg equipment and timers; back day uses another set. Every coach reads the same schedule so one day's timer is not applied to another workout.

## 自己跑一遍 / Try it yourself

```python
warmup = 2
ratio = 3
for it in range(1, 11):
    student = (warmup == -1 or it > warmup) and it % ratio == 0
    active = "net" if student else "fake_score"
    ema = "update" if student and it >= 5 else "skip"
    print(it, active, ema)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
1 fake_score skip
2 fake_score skip
3 net skip
4 fake_score skip
5 fake_score skip
6 net update
...
```

中文: 训练相位、optimizer 和 EMA 都由同一个布尔条件驱动。

English: Training phase, optimizer choice, and EMA all follow the same boolean predicate.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **GAN training loops / GAN training loops**: 中文: 判别器和生成器常按不同频率更新。 / English: Discriminators and generators are often updated at different frequencies.
- **RL actor-critic updates / RL actor-critic updates**: 中文: actor、critic、target network 也需要统一相位控制。 / English: Actor, critic, and target networks also need one phase controller.

## 注意事项 / Caveats / when it breaks

- **ratio 太大 / Ratio too large**: 中文: student 更新太稀疏会让 EMA 和生成器落后。 / English: Sparse student updates can make the EMA and generator lag.
- **hook 顺序 / Hook order**: 中文: EMA 必须在正确的 optimizer step 之后、zero_grad 前后顺序清晰。 / English: EMA must run around the correct optimizer step with clear zero-grad ordering.

## 延伸阅读 / Further reading

- Source permalink above.
- Gamma-World repository: https://github.com/nv-tlabs/Gamma-World
