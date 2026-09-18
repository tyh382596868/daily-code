---
date: 2026-09-18
topic: diffusion
source: trending
repo: meituan-longcat/WBench
file: src/metrics/interaction/navigation_trajectory.py
permalink: https://github.com/meituan-longcat/WBench/blob/48962436d92c6867426e76daf33dbd9484764ce1/src/metrics/interaction/navigation_trajectory.py#L534-L625
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, diffusion, world-model, evaluation, trajectory-metrics]
---

# WBench NavScore：别把“走得远”误当成“走得对” / WBench NavScore: Do Not Confuse Going Far with Going Correctly

> **一句话 / In one line**: WBench 先按动作生成自适应 ground truth，再把 trajectory error 归一化成 accuracy 与 consistency，并在没有可比较动作对时返回未定义。 / WBench builds action-aware ground truth, normalizes trajectory error into accuracy and consistency, and returns undefined when no comparable action pair exists.

## 为什么重要 / Why this matters

交互式世界模型的导航评测不能只看最后一帧像不像。一个模型可能沿着正确方向移动但速度不同，也可能每一步都“看起来合理”却在重复动作之间不一致。WBench 的导航评分把这些问题拆成可解释的几块。

Interactive world-model navigation cannot be evaluated only by whether the final frame looks plausible. A model may move in the right direction at a different speed, or produce locally plausible turns that are inconsistent when the same action is repeated. WBench separates these failure modes.

`evaluate_navigation` 的主线是：按 turn 生成 GT segment，按弧长对齐 predicted/GT poses，用总路径长度和总旋转做归一化，再额外计算同类动作之间的 consistency。最值得注意的是，缺少重复或镜像动作时，consistency 不是 1，而是 `None`。

The main flow is: build a GT segment per turn, arc-resample predicted and GT poses, normalize by total path and rotation, then compute consistency across same-group actions. Crucially, when no repeated or mirrored action pair exists, consistency is `None`, not a perfect score.

## 代码 / The code

`meituan-longcat/WBench` — [`src/metrics/interaction/navigation_trajectory.py`](https://github.com/meituan-longcat/WBench/blob/48962436d92c6867426e76daf33dbd9484764ce1/src/metrics/interaction/navigation_trajectory.py#L534-L625)

```python
def evaluate_navigation(
    poses: np.ndarray,
    turn_bounds: List[Tuple[int, int]],
    actions: List[str],
    perspective: str = "first_person",
) -> Dict[str, float]:
    """
    Evaluate navigation trajectory quality using nATE-based NavScore.
    """
    assert len(turn_bounds) == len(actions)

    R_world = poses[turn_bounds[0][0], :3, :3].copy()
    gt_segments = []
    current_pos = poses[turn_bounds[0][0], :3, 3].copy()

    for (s, e), act in zip(turn_bounds, actions):
        pred_turn = poses[s:e]

        if _is_pure_translation(act):
            gt_seg = _build_gt_translation(pred_turn, act, R_world)
        else:
            gt_seg = _build_gt_rotation(pred_turn, act, perspective)

        offset = current_pos - gt_seg[0, :3, 3]
        gt_seg_global = gt_seg.copy()
        gt_seg_global[:, :3, 3] += offset

        current_pos = gt_seg_global[-1, :3, 3].copy()
        if not _is_pure_translation(act):
            R_world = gt_seg_global[-1, :3, :3].copy()

        gt_segments.append(gt_seg_global)

    gt_sampled_parts, pred_sampled_parts = [], []
    for (s, e), gt_seg in zip(turn_bounds, gt_segments):
        pred_turn = poses[s:e]
        nt = min(len(gt_seg), len(pred_turn))
        gt_sampled_parts.append(_resample_by_arc(gt_seg[:nt], ARC_SAMPLES_PER_TURN))
        pred_sampled_parts.append(_resample_by_arc(pred_turn[:nt], ARC_SAMPLES_PER_TURN))
    gt_sampled = np.concatenate(gt_sampled_parts, axis=0)
    pred_sampled = np.concatenate(pred_sampled_parts, axis=0)
    ate_t, ate_r = _compute_ate(gt_sampled, pred_sampled)

    total_s, total_e = turn_bounds[0][0], turn_bounds[-1][1]
    pred_full = poses[total_s:total_e]
    total_path = _path_length(pred_full)
    total_rot = _total_rotation_deg(pred_full)
    norm_path = max(total_path, MIN_DISP_NORM)
    norm_rot = max(total_rot, MIN_ROT_NORM)

    nate_t = min(ate_t / norm_path, 1.0)
    nate_r = min(ate_r / norm_rot, 1.0)
    ct = _compute_consistency_trajectory(poses, turn_bounds, actions)

    accuracy = 1.0 - (nate_t + nate_r) / 2.0
    if ct["n_pairs"] > 0:
        consistency = 1.0 - (ct["cnATE_t"] + ct["cnATE_r"]) / 2.0
        nav_score = (accuracy + consistency) / 2.0
    else:
        consistency = None
        nav_score = None

    return {
        "NavScore": nav_score,
        "accuracy": float(accuracy),
        "consistency": consistency,
        "nATE_t": float(nate_t),
        "nATE_r": float(nate_r),
        "cnATE_t": ct["cnATE_t"],
        "cnATE_r": ct["cnATE_r"],
        "consistency_pairs": ct["n_pairs"],
        "ATE_t": float(ate_t),
        "ATE_r": float(ate_r),
        "total_path_length": float(total_path),
        "total_rotation_deg": float(total_rot),
    }
```

## 逐行讲解 / What's happening

1. **第 552-574 行 / Lines 552-574 (action-aware GT)**:
   - 中文: 每个 turn 根据动作选择 translation 或 rotation builder，再把局部 GT 对齐到上一段的终点，保证整条轨迹连续。
   - English: Each turn selects a translation or rotation builder, then offsets the local GT to the previous segment's endpoint so the full trajectory stays continuous.
2. **第 576-585 行 / Lines 576-585 (arc-length sampling)**:
   - 中文: 不直接按帧号比较，而是按弧长重采样，让“走得快慢不同”不会单独制造巨大误差。
   - English: The evaluator resamples by arc length instead of frame index, so different traversal speeds do not create a large error by themselves.
3. **第 587-597 行 / Lines 587-597 (normalized ATE)**:
   - 中文: 平移误差除以路径长度，旋转误差除以总旋转角；短轨迹使用下限，避免除以接近零的数。
   - English: Translation error is normalized by path length and rotation error by total rotation, with floors that prevent unstable near-zero division.
4. **第 598-605 行 / Lines 598-605 (consistency)**:
   - 中文: `cnATE` 来自相同或镜像动作之间的轨迹形状比较；有比较对时才与 accuracy 合成 NavScore。
   - English: `cnATE` compares trajectory shape across identical or mirrored actions; only available pairs contribute to the combined NavScore.
5. **第 606-610 行 / Lines 606-610 (undefined is honest)**:
   - 中文: 没有重复动作就没有 consistency 证据，返回 `None` 比伪造满分更诚实，也让 model-level 聚合可以按各自可用 case 计算。
   - English: Without repeated actions there is no consistency evidence. Returning `None` is more honest than inventing a perfect score and lets model-level aggregation use the cases where each component exists.

## 类比 / The analogy

想象评判跑步机器人。只看终点会漏掉“绕了一大圈才回来”；只看每秒位置又会惩罚一个只是速度不同的机器人。WBench 像同时看路线形状、方向动作和重复路线的稳定性。

Imagine judging a running robot. The finish line misses a robot that took a huge detour, while frame-by-frame distance unfairly punishes a robot that simply moved at another speed. WBench checks route shape, action direction, and repeatability together.

## 自己跑一遍 / Try it yourself

```python
def nav_score(ate_t, ate_r, path, rotation, pairs, cnate_t=0.0, cnate_r=0.0):
    nate_t = min(ate_t / max(path, 0.5), 1.0)
    nate_r = min(ate_r / max(rotation, 10.0), 1.0)
    accuracy = 1.0 - (nate_t + nate_r) / 2
    if pairs == 0:
        return {"accuracy": round(accuracy, 3), "consistency": None, "NavScore": None}
    consistency = 1.0 - (cnate_t + cnate_r) / 2
    return {"accuracy": round(accuracy, 3),
            "consistency": round(consistency, 3),
            "NavScore": round((accuracy + consistency) / 2, 3)}

print(nav_score(0.2, 4.0, 2.0, 40.0, 0))
print(nav_score(0.2, 4.0, 2.0, 40.0, 2, 0.1, 0.2))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```
{'accuracy': 0.8, 'consistency': None, 'NavScore': None}
{'accuracy': 0.8, 'consistency': 0.85, 'NavScore': 0.825}
```

中文: 第二个 case 只有在存在两段可比较动作时才得到 NavScore。  
English: The second case gets a NavScore only because comparable action pairs exist.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **World-model benchmark leaderboards** / **World-model benchmark leaderboards**: 中文: 需要把视觉质量、可控性、时间一致性和物理合理性拆开报告。 / English: They need separate reports for visual quality, controllability, temporal consistency, and physical plausibility.
- **Robot trajectory evaluation** / **Robot trajectory evaluation**: 中文: ATE、RPE 和按路径长度归一化的误差同样避免绝对尺度主导分数。 / English: ATE, RPE, and path-normalized errors likewise prevent absolute scale from dominating the score.
- **Action-conditioned WAM evaluation** / **Action-conditioned WAM evaluation**: 中文: action 不是附加标签，而是应当决定目标轨迹与可验证的 motion evidence。 / English: Actions are not mere labels; they should define target motion and verifiable motion evidence.

## 注意事项 / Caveats / when it breaks

- **GT builder 依赖动作词表** / **The GT builder depends on the action vocabulary**: 中文: 新动作如果没有 `ACTION_TO_MOTION` 映射，会无法产生有意义的 reference。 / English: A new action without an `ACTION_TO_MOTION` mapping cannot produce a meaningful reference.
- **弧长采样需要稳定姿态** / **Arc sampling needs valid poses**: 中文: 旋转矩阵退化或轨迹点重复过多时，Slerp 和归一化都可能变得不稳定。 / English: Degenerate rotations or too many repeated points can destabilize Slerp and normalization.
- **None 需要上层理解** / **The caller must understand None**: 中文: 聚合器不能把未定义 consistency 当成 0 或 1，必须按可用样本分别统计。 / English: Aggregators must not coerce undefined consistency to 0 or 1; they should average over defined cases.

## 延伸阅读 / Further reading

- [WBench repository](https://github.com/meituan-longcat/WBench)
- [WBench navigation metric](https://github.com/meituan-longcat/WBench/blob/48962436d92c6867426e76daf33dbd9484764ce1/src/metrics/interaction/navigation_trajectory.py)
- [WBench README and leaderboard](https://github.com/meituan-longcat/WBench#readme)
