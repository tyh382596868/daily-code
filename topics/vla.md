# VLA — build your own

Notes tagged `vla`, newest first. Daily teaching points sourced from
[openvla](https://github.com/openvla/openvla),
[openvla-oft](https://github.com/openvla/openvla-oft),
[lerobot](https://github.com/huggingface/lerobot),
[openpi](https://github.com/Physical-Intelligence/openpi),
[Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T),
and [starVLA](https://github.com/starVLA/starVLA).

Each entry teaches **one component** of a Vision-Language-Action model and
maps it explicitly to its role in a from-scratch `nanoVLA` / production VLA build.

| Date | Component | Title | Repo |
|------|-----------|-------|------|
| 2026-06-09 | vlm-backbone-wiring | [GR00T-N1.7 数据流完整拆解:image / language / state / action 如何经 cross-attention DiT 变成 action / GR00T-N1.7 end-to-end data flow: image / language / state / action turning into action via cross-attention DiT](../nano/vla/2026-06-09-groot-cross-attention-multimodal-fusion.md) | [NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) |
| 2026-06-09 | action-head-continuous | [pi0 完整数据流:image / language / state / action 四模态如何流到最终 action / pi0 end-to-end data flow: how image / language / state / action turn into final action](../nano/vla/2026-06-09-pi0-flow-matching-multimodal-fusion.md) | [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) |
| 2026-06-09 | action-chunking | [Real-Time Chunking:让动作 chunk 之间的接缝消失的 130 行 / Real-Time Chunking: 130 lines that make the seams between action chunks disappear](../nano/vla/2026-06-09-lerobot-rtc-action-chunking.md) | [huggingface/lerobot](https://github.com/huggingface/lerobot) |
| 2026-06-08 | vlm-backbone-wiring | [pi0-FAST 把 state / action / language 全塞进同一条 token 流,靠 PaliGemma 的 prefix-LM mask 完成融合 / pi0-FAST stuffs state / action / language into one token stream and lets PaliGemma's prefix-LM mask do the fusion](../nano/vla/2026-06-08-pi0fast-multimodal-prefix-lm-fusion.md) | [huggingface/lerobot](https://github.com/huggingface/lerobot) |
| 2026-06-08 | inference-loop | [pi0-FAST 怎么知道 action 该停了:训练埋两个 stop signal,JAX 和 PyTorch 各用一个 / How pi0-FAST knows when actions should stop: training plants two stop signals, JAX and PyTorch each pick a different one](../nano/vla/2026-06-08-pi0fast-stop-signals-decode-loop.md) | [huggingface/lerobot](https://github.com/huggingface/lerobot) |
| 2026-06-08 | action-head-continuous | [OpenVLA-OFT 把 LLaMA 退化成"位置查询编码器":action 位置全塞零,L1 head 一次出 8 步 / OpenVLA-OFT turns LLaMA into a "position-only query encoder": zero the action embeddings, let the L1 head emit 8 steps at once](../nano/vla/2026-06-08-openvla-oft-zero-action-l1-head.md) | [openvla/openvla-oft](https://github.com/openvla/openvla-oft) |
| 2026-06-08 | vlm-backbone-wiring | [把要预测的位置塞 placeholder,让 attention 从 context 单向"填空" — 这条设计线从 BERT 走到了 OFT / Put placeholders at positions to be predicted, let attention "fill in" from context — a design lineage from BERT (2018) to OpenVLA-OFT (2025)](../nano/vla/2026-06-08-openvla-oft-placeholder-attention-lineage.md) | [openvla/openvla-oft](https://github.com/openvla/openvla-oft) |
| 2026-06-08 | vlm-backbone-wiring | [OpenVLA 没有"融合模块":vision 钉前缀,action 钉后缀,32 层 causal attention 自己融 / OpenVLA has no "fusion module": vision pinned at prefix, action pinned at suffix, 32 layers of causal attention do the rest](../nano/vla/2026-06-08-openvla-multimodal-fusion-causal-mask.md) | [openvla/openvla](https://github.com/openvla/openvla) |
| 2026-06-08 | training-step | [OpenVLA 的训练目标就是标准 LM 的 next-token prediction,只是 labels 多了一行 mask / OpenVLA's training target is just standard LM next-token prediction — only one line of label masking restricts the loss to the 7 action positions](../nano/vla/2026-06-08-openvla-next-token-prediction-target.md) | [openvla/openvla](https://github.com/openvla/openvla) |
| 2026-06-08 | action-head-continuous | [整个 GR00T 的训练步骤就 6 行干净的 flow-matching / The whole GR00T training step is six clean lines of flow matching](../nano/vla/2026-06-08-lerobot-groot-flow-matching-action-head.md) | [huggingface/lerobot](https://github.com/huggingface/lerobot) |
| 2026-06-08 | vision-encoder | [用一层 Conv2d 把图片切成 token:nanoVLM 的视觉编码器 / One Conv2d turns pixels into tokens: nanoVLM's vision encoder](../nano/vla/2026-06-08-nanovlm-vit-patch-embed.md) | [huggingface/nanoVLM](https://github.com/huggingface/nanoVLM) |
| 2026-06-08 | vision-encoder | [37 行的 ViTPatchEmbeddings:一个 Conv2d 就是整个"图像分块" / 37 lines of ViTPatchEmbeddings: one Conv2d *is* the entire "patchify" step](../nano/vla/2026-06-08-nanovlm-vit-patch-embeddings.md) | [huggingface/nanoVLM](https://github.com/huggingface/nanoVLM) |
| 2026-06-07 | vision-encoder | [一根 stride 等于 patch 的 Conv2d:VLA 视觉编码器的整个入口就这么简单 / One Conv2d with stride = patch size: the entire entry point of a VLA's vision encoder](../nano/vla/2026-06-07-nanovlm-vit-patch-embeddings.md) | [huggingface/nanoVLM](https://github.com/huggingface/nanoVLM) |
| 2026-06-05 | vision-encoder | [一颗 Conv2d 就是 patch embed:从零搭一个能给 VLA 用的 ViT / One Conv2d is your patch embed: a ViT from scratch ready to feed a VLA](../nano/vla/2026-06-05-nanovlm-vit-from-scratch.md) | [huggingface/nanoVLM](https://github.com/huggingface/nanoVLM) |
| 2026-06-04 | vision-encoder | [One Conv2d is the entire patch embedding (nanoVLM's ViTPatchEmbeddings)](../nano/vla/2026-06-04-nanovlm-vit-patch-embed.md) | [huggingface/nanoVLM](https://github.com/huggingface/nanoVLM) |
| 2026-06-03 | vision-encoder | [把 SigLIP 的预训练权重灌进自己的"fused-QKV" ViT / Loading SigLIP's pretrained weights into your own fused-QKV ViT](../nano/vla/2026-06-03-nanovlm-vit-from-pretrained.md) | [huggingface/nanoVLM](https://github.com/huggingface/nanoVLM) |
| 2026-06-02 | nano | [一个 Conv2d 等于整个 ViT 的入口 / One Conv2d *is* the entire ViT entry point](../nano/vla/2026-06-02-nanovlm-vit-patch-embeddings.md) | [huggingface/nanoVLM](https://github.com/huggingface/nanoVLM) |
| 2026-06-01 | vision-encoder | [40 行 ViTPatchEmbeddings:把像素切成 token 的最小实现 / 40-line ViTPatchEmbeddings: the smallest possible patchifier](../nano/vla/2026-06-01-nanovlm-vit-patch-embeddings-v1.md) | [huggingface/nanoVLM](https://github.com/huggingface/nanoVLM) |
| 2026-06-01 | vision-encoder | [nanoVLM 用一个 Conv2d 把图片变成 token / nanoVLM turns an image into tokens with a single Conv2d](../nano/vla/2026-06-01-nanovlm-vit-patch-embeddings-v2.md) | [huggingface/nanoVLM](https://github.com/huggingface/nanoVLM) |
| 2026-05-31 | vision | [nanoVLM 把整个视觉塔写成 52 行 / nanoVLM's entire vision tower fits in 52 lines](../nano/vla/2026-05-31-nanovlm-vit-encoder.md) | [huggingface/nanoVLM](https://github.com/huggingface/nanoVLM) |
| 2026-05-29 | — | [SmolVLA's VLM + slim action expert: deep-copy the config, shrink it, rewire cross-attention](../nano/vla/2026-05-29-smolvla-vlm-with-expert.md) | [huggingface/lerobot](https://github.com/huggingface/lerobot) |
| 2026-05-28 | — | [OpenVLA's training step: 40 lines that supervise a robot policy as if it were an LLM](../nano/vla/2026-05-28-openvla-training-step.md) | [openvla/openvla](https://github.com/openvla/openvla) |
