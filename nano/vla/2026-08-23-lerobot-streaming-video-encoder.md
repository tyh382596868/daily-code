---
date: 2026-08-23
topic: vla
source: vla
repo: huggingface/lerobot
file: src/lerobot/datasets/video_utils.py
permalink: https://github.com/huggingface/lerobot/blob/main/src/lerobot/datasets/video_utils.py#L802-L965
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, vla, short-term-observation-memory, video-encoding]
build_role: short-term-observation-memory advanced variant
---

# LeRobot StreamingVideoEncoder：采集时边录边压 / LeRobot StreamingVideoEncoder: Encode While Recording

> **一句话 / In one line**: LeRobot 为每个相机开一个编码线程，把实时帧送进队列，避免 episode 结束后再批量 PNG 转视频。 / LeRobot starts one encoder thread per camera and feeds frames through queues, avoiding a slow PNG-to-video pass at episode end.

## 为什么重要 / Why this matters

VLA 的短期观测记忆不只发生在模型里，也发生在数据管线里。机器人采集时多相机图像持续到来，如果每一帧先落 PNG，最后再转 MP4，保存 episode 会卡很久。Streaming encoder 把“采集”和“压缩”并行起来。

Short-term observation memory in a VLA is not only a model concern; it is also a data-pipeline concern. During robot collection, many camera frames arrive continuously. Saving every frame as PNG and encoding later makes episode finalization slow. A streaming encoder overlaps collection with compression.

## 代码 / The code

`huggingface/lerobot` — [`src/lerobot/datasets/video_utils.py`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/datasets/video_utils.py#L802-L965)

```python
class StreamingVideoEncoder:
    """Manages per-camera encoder threads for real-time video encoding during recording."""

    def __init__(
        self,
        fps: int,
        rgb_encoder: RGBEncoderConfig | None = None,
        depth_encoder: DepthEncoderConfig | None = None,
        queue_maxsize: int = 30,
        encoder_threads: int | None = None,
    ):
        self.fps = fps
        self._rgb_encoder = rgb_encoder or rgb_encoder_defaults()
        self._depth_encoder = depth_encoder or depth_encoder_defaults()
        self._encoder_threads = encoder_threads
        self.queue_maxsize = queue_maxsize
        self._frame_queues: dict[str, queue.Queue] = {}
        self._result_queues: dict[str, queue.Queue] = {}
        self._threads: dict[str, _CameraEncoderThread] = {}
        self._stop_events: dict[str, threading.Event] = {}
        self._video_paths: dict[str, Path] = {}
        self._dropped_frames: dict[str, int] = {}
        self._episode_active = False
        self._closed = False

    def start_episode(
        self, video_keys: list[str], temp_dir: Path, depth_video_keys: list[str] | None = None
    ) -> None:
        if self._episode_active:
            self.cancel_episode()
        self._dropped_frames.clear()
        if depth_video_keys is None:
            depth_video_keys = []
        for video_key in video_keys:
            frame_queue: queue.Queue = queue.Queue(maxsize=self.queue_maxsize)
            result_queue: queue.Queue = queue.Queue(maxsize=1)
            stop_event = threading.Event()
            temp_video_dir = Path(tempfile.mkdtemp(dir=temp_dir))
            video_path = temp_video_dir / f"{video_key.replace('/', '_')}_streaming.mp4"
            encoder = self._depth_encoder if video_key in depth_video_keys else self._rgb_encoder
            encoder_thread = _CameraEncoderThread(
                video_path=video_path,
                fps=self.fps,
                video_encoder=encoder,
                frame_queue=frame_queue,
                result_queue=result_queue,
                stop_event=stop_event,
                encoder_threads=self._encoder_threads,
            )
            encoder_thread.start()
            self._frame_queues[video_key] = frame_queue
            self._result_queues[video_key] = result_queue
            self._threads[video_key] = encoder_thread
            self._stop_events[video_key] = stop_event
            self._video_paths[video_key] = video_path
        self._episode_active = True

    def feed_frame(self, video_key: str, image: np.ndarray) -> None:
        if not self._episode_active:
            raise RuntimeError("No active episode. Call start_episode() first.")
        thread = self._threads[video_key]
        if not thread.is_alive():
            raise RuntimeError(f"Encoder thread for {video_key} is not alive")
        try:
            self._frame_queues[video_key].put(image.copy(), timeout=0.1)
        except queue.Full:
            self._dropped_frames[video_key] = self._dropped_frames.get(video_key, 0) + 1

    def finish_episode(self) -> dict[str, tuple[Path, dict | None]]:
        if not self._episode_active:
            raise RuntimeError("No active episode to finish.")
        results = {}
        for video_key in self._frame_queues:
            self._frame_queues[video_key].put(None)
        for video_key in self._threads:
            self._threads[video_key].join(timeout=120)
            try:
                status, data = self._result_queues[video_key].get(timeout=5)
                if status == "error":
                    raise RuntimeError(f"Encoder thread for {video_key} failed: {data}")
                results[video_key] = (self._video_paths[video_key], data)
            except queue.Empty:
                results[video_key] = (self._video_paths[video_key], None)
        self._cleanup()
        self._episode_active = False
        return results
```

## 逐行讲解 / What's happening

1. **第 833-845 行 / Lines 833-845**:
   - 中文: encoder 保存每个相机对应的 frame queue、result queue、thread、stop event 和临时视频路径。
   - English: The encoder tracks each camera's frame queue, result queue, thread, stop event, and temporary video path.
2. **第 846-887 行 / Lines 846-887**:
   - 中文: 新 episode 开始时，每个相机都获得独立队列和线程；RGB 与 depth 可以用不同 encoder config。
   - English: At episode start, each camera gets its own queue and thread; RGB and depth streams can use different encoder configs.
3. **第 889-924 行 / Lines 889-924**:
   - 中文: 主采集线程只负责把图像 copy 进队列；队列满了就丢帧并计数，避免采集主循环崩掉。
   - English: The recorder thread only copies frames into the queue. If the queue is full, it drops frames and counts them instead of crashing collection.
4. **第 925-965 行 / Lines 925-965**:
   - 中文: episode 结束时发送 sentinel，等待线程收尾，收集视频路径和统计信息。
   - English: At episode end, sentinel values stop the threads, and the method collects video paths plus optional statistics.

## 类比 / The analogy

这像餐厅后厨的传送带。服务员不断把盘子放上传送带，洗碗工在另一头并行处理。如果洗碗工太慢，传送带满了，系统会记录掉了多少盘，而不是让整个餐厅停摆。

It is like a restaurant conveyor belt. Servers keep placing plates on the belt while dishwashers process them in parallel. If the dishwasher side is too slow, the system records dropped plates instead of stopping the whole restaurant.

## 在 nanoVLA 中的位置 / Where this lives in your nano-VLA

在 nanoVLA 里，这属于 `short-term-observation-memory` 的数据侧实现：它把连续视觉帧稳定保存成可重放的视频序列。上游是机器人相机驱动，下游是 dataset reader、delta timestamp 采样器和视觉 encoder。生产级系统还需要时间戳校验、丢帧报警、磁盘配额和多机上传。

In a nanoVLA, this is the data-side implementation of `short-term-observation-memory`: it turns continuous camera frames into replayable video sequences. Upstream are robot camera drivers; downstream are the dataset reader, delta-timestamp sampler, and vision encoder. A production system also needs timestamp checks, dropped-frame alarms, storage quotas, and upload handling.

## 自己跑一遍 / Try it yourself

```python
from queue import Queue, Full

class ToyStream:
    def __init__(self, maxsize=2):
        self.q = Queue(maxsize=maxsize)
        self.dropped = 0

    def feed(self, frame):
        try:
            self.q.put(frame.copy(), timeout=0.01)
        except Full:
            self.dropped += 1

stream = ToyStream(maxsize=2)
for frame in [[1], [2], [3], [4]]:
    stream.feed(frame)

print(list(stream.q.queue))
print("dropped", stream.dropped)
```

运行 / Run with:

```bash
python try.py
```

预期输出 / Expected output:

```text
[[1], [2]]
dropped 2
```

这个 toy 版本展示了 bounded queue 的取舍：保持主循环活着，但必须监控丢帧。

This toy version shows the bounded-queue tradeoff: keep the main loop alive, but monitor frame drops.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **ROS bag recording** / **ROS bag recording**: 传感器数据先进入队列，再由 writer 落盘。 / Sensor data enters queues before writers persist it.
- **LeRobot delta timestamps** / **LeRobot delta timestamps**: 读取时把多帧历史窗口恢复给 policy。 / The reader reconstructs multi-frame history windows for the policy.
- **生产机器人日志系统** / **Production robot logging systems**: 录制、压缩、上传通常分成不同线程。 / Recording, compression, and upload are usually split across threads.

## 注意事项 / Caveats / when it breaks

- **丢帧不是小事** / **Dropped frames matter**: policy 训练时可能学到错误时间间隔。 / Policy training may see incorrect temporal spacing.
- **copy 有成本** / **Copies cost memory bandwidth**: `image.copy()` 防止 buffer 复用竞态，但会增加开销。 / `image.copy()` prevents buffer reuse races but costs bandwidth.
- **线程失败要显式传播** / **Thread failures must propagate**: 编码线程崩了不能静默吞掉。 / Encoder thread crashes must not be swallowed silently.

## 延伸阅读 / Further reading

- LeRobot video utilities: https://github.com/huggingface/lerobot/blob/main/src/lerobot/datasets/video_utils.py
- LeRobot datasets docs: https://huggingface.co/docs/lerobot/
