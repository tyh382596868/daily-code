---
date: 2026-07-31
topic: diffusion
source: trending
repo: FireRedTeam/FireRed-OpenStoryline
file: src/open_storyline/mcp/sampling_handler.py
permalink: https://github.com/FireRedTeam/FireRed-OpenStoryline/blob/c9e945215586f45c12a61c1951ee9a8e9c43a027/src/open_storyline/mcp/sampling_handler.py#L82-L143
difficulty: beginner
read_time: ~10 min
tags: [code-of-the-day, diffusion, video-agent, sampling]
---

# FireRed OpenStoryline：把视频片段采成 VLM 证据 / FireRed OpenStoryline: Sample a Video Segment into VLM Evidence

> **一句话 / In one line**: `_sample_video_segment_to_data_urls` 把 `[in_sec, out_sec]` 时间窗裁到合法范围，并从桶中心采样若干帧。 / `_sample_video_segment_to_data_urls` clamps a `[in_sec, out_sec]` window and samples frames from bucket centers.

## 为什么重要 / Why this matters

视频 agent 不能把完整视频直接塞进 LLM/VLM。更可靠的做法是把时间窗变成少量代表帧，并附上相对时间戳。FireRed-OpenStoryline 是 2026-07-31 仍在更新的 trending 项目，这段函数正好展示了这种工具层接口。

A video agent cannot dump an entire video into an LLM/VLM. A more reliable pattern is to convert a time window into a small set of representative frames with relative timestamps. FireRed-OpenStoryline is a trending project updated on 2026-07-31, and this function shows that tool boundary clearly.

## 代码 / The code

`FireRedTeam/FireRed-OpenStoryline` — [`src/open_storyline/mcp/sampling_handler.py`](https://github.com/FireRedTeam/FireRed-OpenStoryline/blob/c9e945215586f45c12a61c1951ee9a8e9c43a027/src/open_storyline/mcp/sampling_handler.py#L82-L143)

```python
def _choose_num_frames(duration_sec: float, min_frames: int, max_frames: int, frames_per_sec: float) -> int:
    duration_sec = max(0.0, float(duration_sec))
    n = int(math.ceil(duration_sec * frames_per_sec))
    n = max(min_frames, n)
    n = min(max_frames, n)
    return n

def _sample_video_segment_to_data_urls(
    video_path: str,
    in_sec: float,
    out_sec: float,
    resize_edge: int,
    jpeg_quality: int,
    min_frames: int,
    max_frames: int,
    frames_per_sec: float,
) -> List[Tuple[float, str]]:
    """
    Sample frames only from the [in_sec, out_sec] segment. Returns (rel_t_from_in, data_url)
    """
    in_sec = float(in_sec)
    out_sec = float(out_sec)
    clip = VideoFileClip(video_path, audio=False)
    try:
        vdur = float(clip.duration or 0.0)
        if vdur <= 0:
            t = max(0.0, in_sec)
            frame = clip.get_frame(t)
            img = Image.fromarray(frame)
            return [(0.0, _pil_to_data_url(img, resize_edge, jpeg_quality))]
        in_sec = max(0.0, min(in_sec, vdur))
        out_sec = max(0.0, min(out_sec, vdur))
        if out_sec <= in_sec:
            frame = clip.get_frame(in_sec)
            img = Image.fromarray(frame)
            return [(0.0, _pil_to_data_url(img, resize_edge, jpeg_quality))]
        seg_dur = out_sec - in_sec
        n = _choose_num_frames(seg_dur, min_frames, max_frames, frames_per_sec)
        times = [((i + 0.5) / n) * seg_dur for i in range(n)]
        out: List[Tuple[float, str]] = []
        for rel_t in times:
            abs_t = in_sec + rel_t
            frame = clip.get_frame(abs_t)
            img = Image.fromarray(frame)
            out.append((rel_t, _pil_to_data_url(img, resize_edge, jpeg_quality)))
        return out
    finally:
        clip.close()
```

## 逐行讲解 / What's happening

1. **第 82-87 行 / Lines 82-87 (`_choose_num_frames`)**:
   - 中文: 帧数由片段时长和 FPS 推出，再被 `min_frames` 与 `max_frames` 夹住。
   - English: Frame count is derived from duration and FPS, then clamped by `min_frames` and `max_frames`.
2. **第 107-120 行 / Lines 107-120 (duration and clamping)**:
   - 中文: 先读取视频总时长，再把请求窗口裁到 `[0, duration]` 内。
   - English: The function reads total video duration, then clamps the requested window into `[0, duration]`.
3. **第 122-126 行 / Lines 122-126 (invalid window fallback)**:
   - 中文: 如果裁剪后窗口无效，就退化成在起点取一帧，工具仍然返回可消费结果。
   - English: If the clamped window is invalid, it falls back to one frame at the start so the tool still returns usable evidence.
4. **第 131-139 行 / Lines 131-139 (bucket centers)**:
   - 中文: 采样点落在每个桶的中心，避免正好踩在镜头边界或片段边界。
   - English: Samples are placed at bucket centers, avoiding exact shot or segment boundaries.

## 类比 / The analogy

这像给电影剪辑师做缩略图：不是把每一帧都打印出来，而是在每个时间段中间挑一张最能代表那段的照片。

It is like making contact-sheet thumbnails for an editor: you do not print every frame; you pick one representative still from the middle of each time bucket.

## 自己跑一遍 / Try it yourself

```python
import math

def choose_times(in_sec, out_sec, min_frames=2, max_frames=5, fps=1.5):
    in_sec, out_sec = max(0.0, in_sec), max(0.0, out_sec)
    if out_sec <= in_sec:
        return [0.0]
    seg_dur = out_sec - in_sec
    n = min(max_frames, max(min_frames, math.ceil(seg_dur * fps)))
    return [round(((i + 0.5) / n) * seg_dur, 2) for i in range(n)]

print(choose_times(10, 13))
print(choose_times(5, 5))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[0.3, 0.9, 1.5, 2.1, 2.7]
[0.0]
```

第一行是片段内的相对时间，不是视频绝对时间；这让调用方能把证据和原始窗口重新对齐。

The first line contains relative times inside the segment, not absolute video times; that lets the caller realign evidence with the original window.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Video QA preprocessing** / **Video QA preprocessing**: 多数 VLM pipeline 都会先从视频中抽代表帧，再送入模型。 / Most VLM pipelines first sample representative frames before model input.
- **Shot-aware editing agents** / **Shot-aware editing agents**: 剪辑 agent 需要把长视频拆成短证据块，便于规划和检索。 / Editing agents need to split long videos into short evidence blocks for planning and retrieval.

## 注意事项 / Caveats / when it breaks

- **桶中心不懂剧情** / **Bucket centers do not understand story**: 动作峰值可能刚好落在桶边缘，需要镜头检测或运动分数补强。 / A key action may land near a bucket edge; shot detection or motion scoring can improve this.
- **data URL 有大小成本** / **Data URLs have size cost**: JPEG 质量和 resize edge 会直接影响上下文体积。 / JPEG quality and resize edge directly affect context size.

## 延伸阅读 / Further reading

- [FireRed-OpenStoryline sampling source permalink](https://github.com/FireRedTeam/FireRed-OpenStoryline/blob/c9e945215586f45c12a61c1951ee9a8e9c43a027/src/open_storyline/mcp/sampling_handler.py#L82-L143)
