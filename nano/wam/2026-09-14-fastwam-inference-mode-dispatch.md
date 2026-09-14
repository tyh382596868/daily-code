---
date: 2026-09-14
topic: wam
source: wam
repo: yuantianyuan01/FastWAM
file: src/fastwam/models/wan22/fastwam_optional_idm.py
permalink: https://github.com/yuantianyuan01/FastWAM/blob/7faa71108368fbb3b6885649f112af607427a2d4/src/fastwam/models/wan22/fastwam_optional_idm.py#L9-L120
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, wam, sampler-inference, idm, teacher-forcing, runtime-dispatch]
build_role: sampler-inference (advanced variant, explicit IDM versus first-frame action inference)
---

# FastWAM 推理模式：同一个 action API，两个 conditioning regime / FastWAM Inference Modes: One Action API, Two Conditioning Regimes

> **一句话 / In one line**: FastWAM 在同一个 `infer_action` 入口里切换 IDM teacher-forcing 和 first-frame inference，并为每条路径校验自己的输入契约。 / FastWAM switches between IDM teacher-forcing and first-frame inference behind one `infer_action` entry point, validating each path's input contract.

## 为什么重要 / Why this matters

中文：WAM 的动作推理并不只有一种成本和信息量：完整的视频条件可以更丰富，但代价更高；只用第一帧则更轻量，适合低延迟部署。FastWAM 没有把两套 API 暴露给上层，而是用 `action_infer_mode` 做窄接口分派，并让 `idm` 分支明确要求 `num_video_frames`。这是一种把实验模式变成稳定部署契约的办法。

English: WAM action inference has different cost and information regimes. Full video conditioning can be richer but more expensive, while first-frame inference is lighter and better suited to low-latency deployment. FastWAM keeps one public action API and dispatches with `action_infer_mode`, making the extra `num_video_frames` requirement explicit for IDM.

## 代码 / The code

`yuantianyuan01/FastWAM` — [`src/fastwam/models/wan22/fastwam_optional_idm.py`](https://github.com/yuantianyuan01/FastWAM/blob/7faa71108368fbb3b6885649f112af607427a2d4/src/fastwam/models/wan22/fastwam_optional_idm.py#L9-L120)

```python
class FastWAMOptionalIDM(FastWAMIDM):
    """FastWAM variant where full-video IDM conditioning is optional."""

    action_idm_prob: float

    @classmethod
    def from_wan22_pretrained(cls, *, action_idm_prob: float, **kwargs):
        prob = float(action_idm_prob)
        if not 0.0 <= prob <= 1.0:
            raise ValueError(f"`action_idm_prob` must be in [0, 1], got {prob}.")
        model = super().from_wan22_pretrained(**kwargs)
        model.action_idm_prob = prob
        return model

    @torch.no_grad()
    def _build_teacher_forcing_attention_mask(
        self,
        noisy_video_seq_len: int,
        cond_video_seq_len: int,
        action_seq_len: int,
        noisy_video_tokens_per_frame: int,
        cond_video_tokens_per_frame: int,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        full_cond_mask = super()._build_teacher_forcing_attention_mask(
            noisy_video_seq_len=noisy_video_seq_len,
            cond_video_seq_len=cond_video_seq_len,
            action_seq_len=action_seq_len,
            noisy_video_tokens_per_frame=noisy_video_tokens_per_frame,
            cond_video_tokens_per_frame=cond_video_tokens_per_frame,
            batch_size=batch_size,
            device=device,
        )

        noisy_end = noisy_video_seq_len
        cond_end = noisy_video_seq_len + cond_video_seq_len
        first_frame_tokens = min(cond_video_tokens_per_frame, cond_video_seq_len)
        first_frame_cond_mask = full_cond_mask.clone()
        first_frame_cond_mask[cond_end:, noisy_end:cond_end] = False
        first_frame_cond_mask[
            cond_end:,
            noisy_end : noisy_end + first_frame_tokens,
        ] = True

        use_idm_mask = (
            torch.rand((batch_size, 1, 1, 1), device=device) < self.action_idm_prob
        )
        return torch.where(use_idm_mask, full_cond_mask, first_frame_cond_mask)

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        num_video_frames: Optional[int] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        compile_action_infer: bool = False,
        action_infer_mode: str = "idm",
    ) -> dict[str, Any]:
        if action_infer_mode == "idm":
            if num_video_frames is None:
                raise ValueError("`num_video_frames` is required for `action_infer_mode='idm'`.")
            return FastWAMIDM.infer_action(
                self,
                prompt=prompt,
                input_image=input_image,
                action_horizon=action_horizon,
                num_video_frames=num_video_frames,
                proprio=proprio,
                context=context,
                context_mask=context_mask,
                negative_prompt=negative_prompt,
                text_cfg_scale=text_cfg_scale,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
                tiled=tiled,
                compile_action_infer=compile_action_infer,
            )

        if action_infer_mode == "first_frame":
            return FastWAM.infer_action(
                self,
                prompt=prompt,
                input_image=input_image,
                action_horizon=action_horizon,
                proprio=proprio,
                context=context,
                context_mask=context_mask,
                negative_prompt=negative_prompt,
                text_cfg_scale=text_cfg_scale,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
                tiled=tiled,
                compile_action_infer=compile_action_infer,
            )

        raise ValueError("`action_infer_mode` must be one of: idm, first_frame.")
```

## 逐行讲解 / What's happening

1. **第 9-21 行 / Lines 9-21 (probability contract)**:
   - 中文：构造函数把 `action_idm_prob` 变成 float，并限制在 `[0, 1]`；配置错误在模型加载时暴露，而不是训练半途才爆。
   - English: Construction normalizes `action_idm_prob` to a float and bounds it to `[0, 1]`, surfacing configuration errors at load time.
2. **第 23-57 行 / Lines 23-57 (mask mixture)**:
   - 中文：先复用父类的完整 teacher-forcing mask，再复制一份，把动作 token 对整段条件视频的访问收窄为第一帧 token。
   - English: The method reuses the parent's full teacher-forcing mask, clones it, and narrows action-token access from the full conditioning video to the first frame.
3. **第 54-57 行 / Lines 54-57 (`torch.where`)**:
   - 中文：`action_idm_prob` 不是手写 if，而是 batch 级的随机 mask。一个 batch 内可以按样本选择更强或更轻的 conditioning。
   - English: `action_idm_prob` becomes a batch-shaped random mask rather than a Python branch, allowing per-sample selection of richer or lighter conditioning.
4. **第 60-78 行 / Lines 60-78 (`infer_action` contract)**:
   - 中文：两种模式共享 prompt、image、action horizon、scheduler 等参数，只有 `num_video_frames` 是 IDM 特有的额外契约。
   - English: Both modes share prompt, image, action horizon, and scheduler arguments; only `num_video_frames` is an IDM-specific contract.
5. **第 79-99 行 / Lines 79-99 (`idm` branch)**:
   - 中文：IDM 分支先验证帧数存在，再显式调用 `FastWAMIDM.infer_action`，避免依赖 Python 的隐式 MRO 猜测。
   - English: The IDM branch validates the frame count, then explicitly calls `FastWAMIDM.infer_action` instead of relying on implicit method-resolution behavior.
6. **第 101-118 行 / Lines 101-118 (`first_frame` branch)**:
   - 中文：轻量分支绕过 IDM，落到基础 `FastWAM.infer_action`；对外返回形状仍由同一个 action API 保证。
   - English: The lightweight branch bypasses IDM and calls the base `FastWAM.infer_action`, while the public action contract stays unchanged.
7. **第 120 行 / Line 120 (unknown mode)**:
   - 中文：未知字符串立即报错，防止部署配置中的拼写错误悄悄选择错误模型路径。
   - English: Unknown strings fail immediately, preventing a deployment typo from silently selecting the wrong model path.

## 在 nanoWAM 中的位置 / Where this lives in your nanoWAM

中文：这是 `nanoWAM` 的 sampler-inference 适配层，位于 observation/text conditioning 已经准备好之后、flow-matching action denoise loop 之前。它不负责生成动作本身，而是选择“动作 token 能看到多少视频条件”的 regime，并把两种实现统一成 `infer_action(...) -> {"action": ...}`。省掉它，实验代码会把 IDM 和 first-frame 路径散落在 controller、evaluation 和 serving 脚本里。生产版本还要补上 mode 的配置校验、显存预算、缓存生命周期、随机种子、指标记录和运行时 fallback。

English: In a `nanoWAM`, this is the sampler-inference adapter after observation/text conditioning is prepared and before the flow-matching action denoise loop. It does not generate actions itself; it selects how much video context action tokens may see and normalizes both implementations to `infer_action(...) -> {"action": ...}`. Without it, IDM and first-frame branches leak into controllers, evaluation, and serving scripts. Production code should add config validation, memory budgeting, cache lifetime, seeding, metrics, and runtime fallback.

## 自己跑一遍 / Try it yourself

```python
def infer(mode, frames=None):
    if mode == "idm":
        if frames is None:
            raise ValueError("idm needs frames")
        return f"full-video:{frames}"
    if mode == "first_frame":
        return "first-frame"
    raise ValueError("unknown mode")


print(infer("idm", 5))
print(infer("first_frame"))
try:
    infer("idm")
except ValueError as exc:
    print(type(exc).__name__, exc)
```

运行 / Run with:

```bash
python try.py
```

预期输出 / Expected output:

```text
full-video:5
first-frame
ValueError idm needs frames
```

中文：示例把关键契约缩成三条：模式名决定路径，IDM 必须有帧数，未知模式不能静默降级。

English: The toy preserves the three important contracts: the mode selects the path, IDM requires a frame count, and unknown modes cannot silently fall through.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **FastWAM base infer_action** / **FastWAM base infer_action**: first-frame mode caches visual context and denoises actions against it. / First-frame mode caches visual context and denoises actions against it.
- **FastWAM IDM** / **FastWAM IDM**: teacher-forcing conditions action denoising on a richer video sequence. / Teacher-forcing conditions action denoising on a richer video sequence.
- **Open-Sora CFG branches** / **Open-Sora CFG branches**: one sampler can select conditional and unconditional paths while keeping one output contract. / One sampler can select conditional and unconditional paths while keeping one output contract.

## 注意事项 / Caveats / when it breaks

- **mode 不是性能开关而已** / **A mode is more than a performance flag**: 它改变 attention mask 和可见信息，必须记录在实验元数据里。 / It changes the attention mask and visible information, so it belongs in experiment metadata.
- **随机 mask 会影响复现** / **Random masks affect reproducibility**: 训练和评估要明确 seed 与 batch shape。 / Training and evaluation need explicit seeds and batch shapes.
- **统一 API 不代表统一成本** / **One API does not mean one cost**: serving 层仍应按 mode 估算延迟和显存。 / Serving must still budget latency and memory per mode.

## 延伸阅读 / Further reading

- [FastWAM optional IDM](https://github.com/yuantianyuan01/FastWAM/blob/7faa71108368fbb3b6885649f112af607427a2d4/src/fastwam/models/wan22/fastwam_optional_idm.py)
- [FastWAM action-only inference](https://github.com/yuantianyuan01/FastWAM/blob/7faa71108368fbb3b6885649f112af607427a2d4/src/fastwam/models/wan22/fastwam.py#L979-L1159)
