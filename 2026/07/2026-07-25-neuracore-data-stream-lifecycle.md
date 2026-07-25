---
date: 2026-07-25
topic: robotics
source: trending
repo: NeuracoreAI/neuracore
file: neuracore/core/streaming/data_stream.py
permalink: https://github.com/NeuracoreAI/neuracore/blob/ebd04d74e38fbb3c3ac148cc1f1a430a4a7c019d/neuracore/core/streaming/data_stream.py#L83-L180
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, robotics, streaming]
---

# Neuracore DataStream：把机器人传感器流做成可停止的生命周期 / Neuracore DataStream: Give Robot Sensor Streams a Stoppable Lifecycle

> **一句话 / In one line**: 机器人数据流要能 start、prepare stop、drain、stop, 否则上传端很容易丢尾帧。 / Robot data streams need start, prepare-stop, drain, and stop phases, or uploaders easily lose tail frames.

## 为什么重要 / Why this matters

真实机器人采集不是简单 `append(frame)`。视频、关节、点云都在并发写入, 停止录制时还要告诉 producer 不再接收新数据, 同时等待已排队数据 drain。这个 `DataStream` 基类把生命周期拆成清晰的阶段, 还兼容旧 Python daemon 和新 Rust daemon。

Real robot collection is not a simple `append(frame)`. Video, joints, and point clouds are written concurrently, and stopping a recording must reject new data while draining queued data. This `DataStream` base class splits that lifecycle into phases and supports both the legacy Python daemon and the newer Rust daemon path.

## 代码 / The code

`NeuracoreAI/neuracore` -- [`neuracore/core/streaming/data_stream.py`](https://github.com/NeuracoreAI/neuracore/blob/ebd04d74e38fbb3c3ac148cc1f1a430a4a7c019d/neuracore/core/streaming/data_stream.py#L83-L180)

```python
    def start_recording(self, context: DataRecordingContext) -> None:
        """Start recording data for this stream.

        If the stream is already recording, stop it first. Then set the
        recording state to True and store the recording context.

        Args:
            context: Recording context containing identifiers for the recording
                session, robot, and dataset.
        """
        if self.is_recording():
            _, stop_cutoff_sequence_number = self.prepare_recording_stopped()
            self.stop_recording(
                wait_for_producer_drain=False,
                stop_cutoff_sequence_number=stop_cutoff_sequence_number,
            )
        self._recording = True
        self._last_logged_timestamp = None
        self._context = context
        self._handle_ensure_producer_channel(context)

    def _handle_ensure_producer_channel(self, context: DataRecordingContext) -> None:
        """Ensure a legacy producer channel exists for this data stream.

        Under the Rust daemon the stream owns no channel — lifecycle and data
        envelopes are published by ``RecordingContext`` from the logging layer
        — so this is a no-op and ``_producer_channel`` stays ``None``.

        Args:
            context: Recording context containing identifiers for
                the recording session, robot, and dataset.
        """
        if self._use_data_bridge:
            return
        if self._producer_channel is None:
            channel_id = f"{self._data_type.value}:\
            {self._stream_name}:{uuid.uuid4().hex[:8]}"
            self._producer_channel = ProducerChannel(
                id=channel_id,
                recording_id=context.recording_id,
                data_type=self._data_type,
            )

        self._producer_channel.start_recording_session(
            recording_id=context.recording_id
        )

    def prepare_recording_stopped(self) -> tuple[ProducerChannel | None, int]:
        """Mark the producer channel as stopping and return it.

        Under the Rust daemon there is no channel, so this returns ``(None, 0)``.
        Under the legacy daemon a missing channel means the stream is stale, so
        it raises :class:`MissingProducerChannelError` to have it pruned.
        """
        producer_channel = self.get_producer_channel()
        if producer_channel is None:
            if not is_rust_daemon_enabled():
                raise MissingProducerChannelError(
                    "stream has no active producer channel"
                )
            return None, 0

        stop_cutoff_sequence_number = producer_channel.mark_recording_stop_requested()

        return producer_channel, stop_cutoff_sequence_number

    def stop_recording(
        self,
        stop_cutoff_sequence_number: int,
        wait_for_producer_drain: bool = True,
    ) -> None:
        """Stop recording data and tear down the active producer, if any."""
        self._recording = False
        self._context = None
        producer_channel = self._producer_channel
        self._producer_channel = None

        if producer_channel is None:
            if not is_rust_daemon_enabled():
                # Legacy daemon: a stream with no producer channel is stale —
                # raise so the caller prunes it.
                raise MissingProducerChannelError(
                    "stream has no active producer channel"
                )
            # Rust daemon: the stream never owned a channel — nothing to drain.
            return

        try:
            if producer_channel.trace_id:
                producer_channel.cleanup_producer_channel(
                    stop_cutoff_sequence_number=stop_cutoff_sequence_number,
                    wait_for_slot_drain=wait_for_producer_drain,
                )
        finally:
            producer_channel.stop_producer_channel(
                wait_for_slot_drain=wait_for_producer_drain,
            )

```

## 逐行讲解 / What's happening

1. **第 83-102 行 / Lines 83-102 (`start_recording`)**:
   - 中文: 如果已经在录制, 先准备并停止旧 session, 再设置新 context。
   - English: If already recording, it prepares and stops the old session before installing the new context.
2. **第 104-128 行 / Lines 104-128 (`ensure producer`)**:
   - 中文: Rust daemon 路径不建 channel; legacy 路径才创建 ProducerChannel 并开始 session。
   - English: The Rust daemon path owns no channel; the legacy path creates a `ProducerChannel` and starts a session.
3. **第 130-147 行 / Lines 130-147 (`prepare stop`)**:
   - 中文: 先 mark stop requested, 返回 cutoff sequence number 给上层等待 drain。
   - English: It marks stop requested and returns a cutoff sequence number so callers can wait for drain.
4. **第 149-180 行 / Lines 149-180 (`stop_recording`)**:
   - 中文: 清本地状态, 再 cleanup 和 stop producer channel。
   - English: It clears local state, then cleans up and stops the producer channel.

## 类比 / The analogy

这像关闭工厂流水线。先挂出停止接单牌, 再等传送带上已经生产的零件走完, 最后关机器。

It is like shutting down a factory line. First stop accepting new orders, then let parts already on the belt finish, and only then turn off the machine.

## 自己跑一遍 / Try it yourself

```python
class Channel:
    def __init__(self): self.next = 1; self.stopped_at = None
    def mark_recording_stop_requested(self):
        self.stopped_at = self.next - 1
        return self.stopped_at
    def cleanup_producer_channel(self, stop_cutoff_sequence_number, wait_for_slot_drain=True):
        print("drain", stop_cutoff_sequence_number, wait_for_slot_drain)
    def stop_producer_channel(self, wait_for_slot_drain=True):
        print("stop", wait_for_slot_drain)

class Stream:
    def __init__(self): self.recording=False; self.channel=Channel()
    def start(self): self.recording=True
    def prepare_stop(self): return self.channel, self.channel.mark_recording_stop_requested()
    def stop(self, cutoff):
        ch, self.channel, self.recording = self.channel, None, False
        ch.cleanup_producer_channel(cutoff); ch.stop_producer_channel()

s = Stream(); s.start(); _, cutoff = s.prepare_stop(); s.stop(cutoff)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```
drain 0 True
stop True
```

停止不是一个瞬间动作, 而是先切断入口、再 drain、最后释放 channel。

Stopping is not instantaneous. It closes intake, drains, then releases the channel.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **ROS bag recording**: 录制结束也要 flush writer, 否则尾部消息可能丢失。 / ROS bag recording also flushes writers on stop to avoid losing tail messages.
- **视频编码器**: 编码线程通常需要显式 drain 才能写完最后 GOP。 / Video encoders usually need an explicit drain to finish the last GOP.

## 注意事项 / Caveats / when it breaks

- **双 daemon 路径要测试**: Rust 和 legacy 行为不同, 单测要覆盖两边。 / Rust and legacy paths differ, so tests need both.
- **stop 失败要可恢复**: 清本地状态和停止 producer 的顺序会影响重试策略。 / The order of clearing local state and stopping producers affects retries.

## 延伸阅读 / Further reading

- Neuracore repository: https://github.com/NeuracoreAI/neuracore
- GitHub search result showed 274 stars and pushed_at 2026-07-25T08:05:21Z
