---
date: 2026-09-01
topic: robotics
source: tracked
repo: huggingface/lerobot
file: src/lerobot/transport/utils.py
permalink: https://github.com/huggingface/lerobot/blob/d36d404b65315139b7601a707f260a3db736462f/src/lerobot/transport/utils.py#L46-L111
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, robotics, transport, chunking, grpc]
---

# LeRobot chunked transport：大消息拆小块传 / LeRobot Chunked Transport: Send Big Messages in Small Pieces

> **一句话 / In one line**: 这段工具把大块二进制消息切成带状态标记的小块，接收端再按 `BEGIN/MIDDLE/END` 重新拼回完整 payload。 / This utility splits a large binary payload into state-tagged chunks and lets the receiver rebuild it from `BEGIN/MIDDLE/END`.

## 为什么重要 / Why this matters

机器人系统经常要在 actor、learner、policy server 之间传模型权重、交互轨迹或视频证据。一次性塞进 RPC 消息会撞上大小限制，也会让失败恢复变得含糊。LeRobot 这里把传输协议做成很小的状态机：发送端只负责切块和标记，接收端只负责按标记清空、追加、提交。

Robot systems often move model state, interaction batches, or video evidence between actors, learners, and policy servers. A single RPC payload can hit message-size limits and make failures hard to localize. LeRobot keeps the transport as a small state machine: the sender chunks and marks, while the receiver resets, appends, and commits.

## 代码 / The code

`huggingface/lerobot` — [`src/lerobot/transport/utils.py`](https://github.com/huggingface/lerobot/blob/d36d404b65315139b7601a707f260a3db736462f/src/lerobot/transport/utils.py#L46-L111)

```python
def send_bytes_in_chunks(buffer: bytes, message_class: Any, log_prefix: str = "", silent: bool = True):
    bytes_buffer: io.BytesIO = io.BytesIO(buffer)
    size_in_bytes = bytes_buffer_size(bytes_buffer)

    sent_bytes = 0

    logging_method = logging.info if not silent else logging.debug

    logging_method(f"{log_prefix} Buffer size {size_in_bytes / 1024 / 1024} MB with")

    while sent_bytes < size_in_bytes:
        transfer_state = TransferState.TRANSFER_MIDDLE

        if sent_bytes + CHUNK_SIZE >= size_in_bytes:
            transfer_state = TransferState.TRANSFER_END
        elif sent_bytes == 0:
            transfer_state = TransferState.TRANSFER_BEGIN

        size_to_read = min(CHUNK_SIZE, size_in_bytes - sent_bytes)
        chunk = bytes_buffer.read(size_to_read)

        yield message_class(transfer_state=transfer_state, data=chunk)
        sent_bytes += size_to_read
        logging_method(f"{log_prefix} Sent {sent_bytes}/{size_in_bytes} bytes with state {transfer_state}")

    logging_method(f"{log_prefix} Published {sent_bytes / 1024 / 1024} MB")


def receive_bytes_in_chunks(iterator, queue: Queue | None, shutdown_event: MpEvent, log_prefix: str = ""):
    bytes_buffer = io.BytesIO()
    step = 0

    logging.info(f"{log_prefix} Starting receiver")
    for item in iterator:
        logging.debug(f"{log_prefix} Received item")
        if shutdown_event.is_set():
            logging.info(f"{log_prefix} Shutting down receiver")
            return

        if item.transfer_state == TransferState.TRANSFER_BEGIN:
            bytes_buffer.seek(0)
            bytes_buffer.truncate(0)
            bytes_buffer.write(item.data)
            logging.debug(f"{log_prefix} Received data at step 0")
            step = 0
        elif item.transfer_state == TransferState.TRANSFER_MIDDLE:
            bytes_buffer.write(item.data)
            step += 1
            logging.debug(f"{log_prefix} Received data at step {step}")
        elif item.transfer_state == TransferState.TRANSFER_END:
            bytes_buffer.write(item.data)
            logging.debug(f"{log_prefix} Received data at step end size {bytes_buffer_size(bytes_buffer)}")

            if queue is not None:
                queue.put(bytes_buffer.getvalue())
            else:
                return bytes_buffer.getvalue()

            bytes_buffer.seek(0)
            bytes_buffer.truncate(0)
            step = 0

            logging.debug(f"{log_prefix} Queue updated")
        else:
            logging.warning(f"{log_prefix} Received unknown transfer state {item.transfer_state}")
            raise ValueError(f"Received unknown transfer state {item.transfer_state}")
```

## 逐行讲解 / What's happening

1. **第 46-55 行 / Lines 46-55 (`BytesIO` setup)**:
   - 中文: 原始 `bytes` 被包成可顺序读取的缓冲区，同时记录总长度，后面每次只读一段。
   - English: Raw `bytes` are wrapped in a sequential buffer, and the total size is recorded so each loop reads only one slice.
2. **第 56-68 行 / Lines 56-68 (state-tagged chunks)**:
   - 中文: 第一个块标 `TRANSFER_BEGIN`，最后一个块标 `TRANSFER_END`，中间块标 `TRANSFER_MIDDLE`；数据本身不需要知道自己在第几块。
   - English: The first chunk is marked `TRANSFER_BEGIN`, the last `TRANSFER_END`, and all middle chunks `TRANSFER_MIDDLE`; the payload itself does not need embedded offsets.
3. **第 74-90 行 / Lines 74-90 (receiver reset)**:
   - 中文: 收到 `BEGIN` 时先清空缓冲区，避免上一条半包消息污染下一条消息。
   - English: On `BEGIN`, the receiver clears the buffer so a partial previous transfer cannot contaminate the next message.
4. **第 95-106 行 / Lines 95-106 (commit and clear)**:
   - 中文: 收到 `END` 后才把完整 payload 放进队列或返回，然后立刻重置，准备接下一条。
   - English: Only `END` commits the full payload to the queue or return value; the buffer is then reset for the next transfer.

## 类比 / The analogy

像把一套厚说明书分成几个包裹寄出。第一个包写“开始”，中间包写“续件”，最后一个包写“完结”。收件人看到“开始”就清空桌面，看到“完结”才装订成册。

It is like mailing a thick manual in several parcels. The first says "start", the middle ones say "continued", and the last says "complete". The recipient clears the desk on "start" and binds the manual only on "complete".

## 自己跑一遍 / Try it yourself

```python
from dataclasses import dataclass

CHUNK = 4

@dataclass
class Msg:
    state: str
    data: bytes

def send(payload):
    sent = 0
    while sent < len(payload):
        state = "MIDDLE"
        if sent == 0:
            state = "BEGIN"
        if sent + CHUNK >= len(payload):
            state = "END" if sent else "BEGIN_END"
        part = payload[sent:sent + CHUNK]
        sent += len(part)
        yield Msg(state, part)

buf = bytearray()
for msg in send(b"robot-policy-state"):
    if msg.state in {"BEGIN", "BEGIN_END"}:
        buf.clear()
    buf.extend(msg.data)
    if msg.state in {"END", "BEGIN_END"}:
        print(buf.decode())
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
robot-policy-state
```

中文: 重点不是切块大小，而是边界状态让接收端知道什么时候清空、什么时候提交。

English: The key is not the chunk size; it is the boundary state that tells the receiver when to reset and when to commit.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **gRPC streaming** / **gRPC streaming**: 中文: 大响应常用流式消息分块传输，服务端逐块 yield。 / English: Large responses are often streamed as many yielded messages.
- **dataset uploaders** / **Dataset uploaders**: 中文: 大文件上传会把 payload 分片，并在最后一步做完整性确认。 / English: Large-file uploaders split payloads and confirm completeness at the final step.

## 注意事项 / Caveats / when it breaks

- **缺少序号 / No sequence number**: 中文: 这段代码假设底层 stream 保序；如果通道可能乱序，消息还需要 chunk index。 / English: This assumes the stream preserves order; unordered transports need chunk indices.
- **异常中断 / Interrupted transfer**: 中文: 中间断开时接收端只保留半包缓冲，需要上层重试或丢弃。 / English: If the transfer breaks mid-stream, the receiver only has a partial buffer; retries or discard logic must live above it.

## 延伸阅读 / Further reading

- LeRobot transport source: https://github.com/huggingface/lerobot/blob/d36d404b65315139b7601a707f260a3db736462f/src/lerobot/transport/utils.py
