---
date: 2026-09-01
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/utils/data/dataloader.py
permalink: https://github.com/pytorch/pytorch/blob/d47079b9af35a44e41096b04cebad64c0caa89f5/torch/utils/data/dataloader.py#L1470-L1568
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, pytorch, dataloader, prefetch, multiprocessing]
---

# PyTorch DataLoader prefetch：边取边补任务 / PyTorch DataLoader Prefetch: Refill Work While Reading Results

> **一句话 / In one line**: 多进程 DataLoader 一边按顺序交付 batch，一边用 `_try_put_index()` 给 worker 补新任务。 / The multiprocessing DataLoader returns batches while `_try_put_index()` keeps workers stocked with new indices.

## 为什么重要 / Why this matters

训练吞吐经常卡在数据加载，而不是模型 forward。PyTorch 的多 worker DataLoader 不能只“派一批、等一批”，否则 GPU 会空等。这里的核心是维护一组还没返回的 task：结果可以乱序到达，但用户看到的迭代器仍然能按配置顺序产出，同时每消费一个结果就补发一个新索引。

Training throughput is often limited by input loading rather than model compute. A multi-worker DataLoader cannot simply "send one batch, wait for one batch" without starving the GPU. This code tracks outstanding tasks: results may arrive out of order, but the iterator can still yield in configured order and refill one new index for each consumed result.

## 代码 / The code

`pytorch/pytorch` — [`torch/utils/data/dataloader.py`](https://github.com/pytorch/pytorch/blob/d47079b9af35a44e41096b04cebad64c0caa89f5/torch/utils/data/dataloader.py#L1470-L1568)

```python
def _next_data(self):
    while True:
        # If the worker responsible for `self._rcvd_idx` has already ended
        # and was unable to fulfill this task (due to exhausting an `IterableDataset`),
        # we try to advance `self._rcvd_idx` to find the next valid index.
        #
        # This part needs to run in the loop because both the `self._get_data()`
        # call and `_IterableDatasetStopIteration` check below can mark
        # extra worker(s) as dead.
        while self._rcvd_idx < self._send_idx:
            info = self._task_info.get(self._rcvd_idx, None)
            if info:
                worker_id = info[0]
                if (
                    len(info) == 2 or self._workers_status[worker_id]
                ):  # has data or is still active
                    break
                del self._task_info[self._rcvd_idx]
            self._rcvd_idx += 1
        else:
            # no valid `self._rcvd_idx` is found (i.e., didn't break)
            if not self._persistent_workers:
                self._shutdown_workers()
            raise StopIteration

        # Now `self._rcvd_idx` is the batch index we want to fetch

        # Check if the next sample has already been generated
        if len(self._task_info[self._rcvd_idx]) == 2:
            worker_id, data = self._task_info.pop(self._rcvd_idx)
            self._rcvd_idx += 1
            return self._process_data(data, worker_id)

        if self._shutdown or self._tasks_outstanding <= 0:
            raise AssertionError(
                "Invalid iterator state: shutdown or no outstanding tasks when fetching next data"
            )
        idx, data = self._get_data()
        self._tasks_outstanding -= 1
        if self._dataset_kind == _DatasetKind.Iterable:
            # Check for _IterableDatasetStopIteration
            if isinstance(data, _utils.worker._IterableDatasetStopIteration):
                if self._persistent_workers:
                    self._workers_status[data.worker_id] = False
                else:
                    self._mark_worker_as_unavailable(data.worker_id)
                self._try_put_index()
                continue

        if idx != self._rcvd_idx:
            if not self._in_order:
                # don't store it for later, process now
                # delete from self._task_info immediately
                # this keeps the object size manageable
                worker_id = self._task_info.pop(idx)[0]
                return self._process_data(data, worker_id)
            # store out-of-order samples
            self._task_info[idx] += (data,)
        else:
            worker_id = self._task_info.pop(idx)[0]
            self._rcvd_idx += 1
            return self._process_data(data, worker_id)

def _try_put_index(self) -> None:
    max_tasks = self._prefetch_factor * self._num_workers
    if self._tasks_outstanding >= max_tasks:
        raise AssertionError(
            "Number of outstanding tasks exceeded maximum allowed tasks"
        )

    try:
        index = self._next_index()
    except StopIteration:
        return
    for _ in range(self._num_workers):  # find the next active worker, if any
        worker_queue_idx = next(self._worker_queue_idx_cycle)
        if self._workers_status[worker_queue_idx]:
            if self._in_order:
                break
            elif self._workers_num_tasks[worker_queue_idx] < max_tasks // sum(
                self._workers_status
            ):
                # when self._in_order is False, distribute work to a worker if it has capacity
                # _workers_status is updated only in this thread, so the sum is guaranteed > 0
                break
    else:
        # not found (i.e., didn't break)
        return

    self._index_queues[worker_queue_idx].put((self._send_idx, index))  # type: ignore[possibly-undefined]
    self._task_info[self._send_idx] = (worker_queue_idx,)
    self._workers_num_tasks[worker_queue_idx] += 1
    self._tasks_outstanding += 1
    self._send_idx += 1
```

## 逐行讲解 / What's happening

1. **第 1479-1493 行 / Lines 1479-1493 (find the next valid task)**:
   - 中文: `_rcvd_idx` 是用户下一步应该看到的 batch 编号；如果对应 worker 已经耗尽，就跳到下一个可交付编号。
   - English: `_rcvd_idx` is the next batch number the user should see; if its worker is exhausted, the iterator advances to the next deliverable number.
2. **第 1497-1501 行 / Lines 1497-1501 (cached out-of-order data)**:
   - 中文: 如果目标 batch 早就到了，直接从 `_task_info` 里取出返回。
   - English: If the desired batch already arrived earlier, it is returned directly from `_task_info`.
3. **第 1507-1517 行 / Lines 1507-1517 (read and refill on stop)**:
   - 中文: 每收到一个 worker 结果就减少 outstanding 计数；IterableDataset 某个 worker 结束时，会标记状态并尝试补新任务。
   - English: Each received worker result decrements the outstanding count; when an IterableDataset worker ends, its status is updated and a new task is attempted.
4. **第 1519-1531 行 / Lines 1519-1531 (ordered vs unordered delivery)**:
   - 中文: 如果结果不是当前要交付的编号，`in_order=True` 时先缓存，`in_order=False` 时可以直接返回。
   - English: If the result is not the current target index, `in_order=True` stores it, while `in_order=False` may return it immediately.
5. **第 1533-1563 行 / Lines 1533-1563 (prefetch budget)**:
   - 中文: `_try_put_index()` 用 `prefetch_factor * num_workers` 限制飞行中的任务数，并把新索引投递给有容量的 worker。
   - English: `_try_put_index()` caps in-flight tasks at `prefetch_factor * num_workers` and dispatches a new index to a worker with capacity.

## 类比 / The analogy

像厨房有多个厨师，服务员按桌号上菜。菜可能不是按桌号做完；如果餐厅要求按桌号上，服务员就把提前做好的菜放到保温台，同时继续给空闲厨师派下一张单。

It is like a kitchen with several cooks while the waiter serves table numbers in order. Dishes may finish out of order; if service must stay ordered, early dishes wait under heat lamps while the waiter keeps assigning new tickets.

## 自己跑一遍 / Try it yourself

```python
arrivals = [(1, "b1"), (0, "b0"), (3, "b3"), (2, "b2")]
waiting = {}
next_idx = 0

for idx, data in arrivals:
    waiting[idx] = data
    while next_idx in waiting:
        print("yield", waiting.pop(next_idx))
        next_idx += 1
        print("prefetch next index")
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
yield b0
prefetch next index
yield b1
prefetch next index
yield b2
prefetch next index
yield b3
prefetch next index
```

中文: `b1` 先到也不会先交付；它被缓存到 `b0` 到达之后再一起排出。

English: `b1` arrives first but is not yielded first; it waits until `b0` arrives and the ordered stream can advance.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **CPU/GPU pipeline** / **CPU/GPU pipelines**: 中文: 训练循环也会保持若干个 batch 在准备中，减少加速器等待。 / English: Training loops also keep several batches in preparation to reduce accelerator idle time.
- **RPC worker pools** / **RPC worker pools**: 中文: 任务分发器常用 outstanding budget 防止队列无限膨胀。 / English: Task dispatchers often cap outstanding work so queues cannot grow without bound.

## 注意事项 / Caveats / when it breaks

- **内存换吞吐 / Memory trades for throughput**: 中文: `prefetch_factor` 越大，吞吐可能越好，但缓存 batch 占用也更高。 / English: Larger `prefetch_factor` can improve throughput, but cached batches consume more memory.
- **IterableDataset 结束不等于 worker 死亡 / IterableDataset exhaustion is not worker death**: 中文: persistent worker 下一轮 epoch 还能复用，所以状态要区分“本轮没活”和“进程挂了”。 / English: Persistent workers can be reused next epoch, so the state separates "done for this epoch" from "process died".

## 延伸阅读 / Further reading

- PyTorch DataLoader source: https://github.com/pytorch/pytorch/blob/d47079b9af35a44e41096b04cebad64c0caa89f5/torch/utils/data/dataloader.py
