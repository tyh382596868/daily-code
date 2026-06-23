# Robotics

Notes tagged `robotics`, newest first. Covers VLA, manipulation policies, locomotion controllers, sim-to-real, and related code.

| Date | Title | Repo |
|------|-------|------|
| 2026-06-23 | [可组合 VLA 数据变换管道：冻结 dataclass + z-score / 分位数归一化 / Composable VLA Data-Transform Pipeline: Frozen Dataclass + Z-score / Quantile Normalization](../2026/06/2026-06-23-openpi-normalize-transform.md) | [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) |
| 2026-06-23 | [MemoryVLA CogMemBank：ICLR 2026 认知记忆库 — 跨时间 Transformer 检索 + Gate 融合 + ToMe 整合 / MemoryVLA CogMemBank: ICLR 2026 Cognitive Memory Bank — Cross-Transformer Retrieval + GateFusion + ToMe Consolidation](../2026/06/2026-06-23-memoryvla-cogmembank.md) | [shihao1895/MemoryVLA](https://github.com/shihao1895/MemoryVLA) |
| 2026-06-15 | [一个 meta device 怎么悄悄毁掉了 VLA 的 flow-matching 时间采样器 / How `torch.device("meta")` silently destroyed a VLA's flow-matching time sampler](../2026/06/2026-06-15-isaac-groot-n1d7-meta-device-beta-fix.md) | [NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) |
| 2026-06-15 | [NVIDIA Lyra 的 Gumbel-Softmax 直通 top-k:200 行代码搞定"训练时可微、推理时硬 top-k" / NVIDIA Lyra's Gumbel-Softmax straight-through top-k: 200 lines that switch between differentiable training and hard inference](../2026/06/2026-06-15-lyra-gumbel-softmax-token-pruning.md) | [nv-tlabs/lyra](https://github.com/nv-tlabs/lyra) |
| 2026-06-12 | [53 行的 FiLM 残差块 —— Diffusion Policy 的"条件注入"全在这 / 53 lines of FiLM residual block — the whole "conditional injection" of Diffusion Policy lives here](../2026/06/2026-06-12-diffusion-policy-film-residual-block.md) | [real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy) |
| 2026-06-12 | [把"人手当机器人末端执行器"那篇 VITRA:50 行代码搞定异构动作的 masked diffusion loss / VITRA — the "human hand as a robot end-effector" paper: 50 lines handle masked diffusion loss for a heterogeneous action vector](../2026/06/2026-06-12-vitra-masked-multi-component-diffusion-loss.md) | [microsoft/VITRA](https://github.com/microsoft/VITRA) |
| 2026-06-09 | [ROBOMETER 用 11 行就把"离散桶 logits"变成连续进度信号 / ROBOMETER turns "discrete-bin logits" into a continuous progress signal in 11 lines](../2026/06/2026-06-09-robometer-bins-to-continuous.md) | [huggingface/lerobot](https://github.com/huggingface/lerobot) |
| 2026-06-09 | [MemVLA 把 ToMe 搬到了"时间维":VLA 长记忆 bank 的 24 行精华 / MemVLA brings ToMe to the *time* axis: 24 lines of long-memory bank for VLAs](../2026/06/2026-06-09-dexbotic-memvla-token-merge-memory.md) | [dexmal/dexbotic](https://github.com/dexmal/dexbotic) |
| 2026-06-04 | [mjlab uses four boolean masks to train four locomotion modes at once](../2026/06/2026-06-04-mjlab-velocity-command-resample.md) | [mujocolab/mjlab](https://github.com/mujocolab/mjlab) |
| 2026-05-29 | [One Linear layer for every robot body (CategorySpecificLinear)](../2026/05/2026-05-29-isaac-groot-category-specific-linear.md) | [NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) |
| 2026-05-26 | [ReinFlow: rectified flow refactored into PyTorch modules](../2026/05/2026-05-26-reinflow-rectified-flow.md) | [ReinFlow/ReinFlow](https://github.com/ReinFlow/ReinFlow) |
| 2026-05-26 | [π₀'s flow matching loss in 25 lines](../2026/05/2026-05-26-openpi-flow-matching-loss.md) | [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) |
| 2026-05-10 | [OpenVLA action tokenizer](../2026/05/2026-05-10-openvla-action-tokenizer-example.md) | [openvla/openvla](https://github.com/openvla/openvla) |
