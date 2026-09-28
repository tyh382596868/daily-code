---
date: 2026-09-28
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/_inductor/codegen/memory_planning.py
permalink: https://github.com/pytorch/pytorch/blob/2e9b4aff8d49b22bbebf288ccbf63983c51e45f0/torch/_inductor/codegen/memory_planning.py#L35-L132
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, pytorch, memory-planning]
---

# PyTorch Inductor live ranges：不重叠的 tensor 才能共用内存 / PyTorch Inductor Live Ranges: Share Memory Only When Lifetimes Do Not Overlap

> **一句话 / In one line**: 编译器复用 buffer 的第一步，是把每个 tensor 的生命期变成可比较的区间。 / The first step in compiler buffer reuse is turning every tensor lifetime into comparable intervals.

## 为什么重要 / Why this matters

Inductor 要把 eager 程序变成高效 kernel 序列，临时 tensor 如果都单独分配会浪费显存。这里的 `LiveRange` / `LiveRanges` 是内存规划的基本语言：什么时候活着、区间能否合并、两个对象是否重叠。

Inductor lowers eager programs into efficient kernel sequences. If every temporary tensor gets its own allocation, GPU memory is wasted. `LiveRange` and `LiveRanges` are the planner's basic vocabulary: when an object is live, when ranges merge, and whether two objects overlap.

## 代码 / The code

`pytorch/pytorch` — [`torch/_inductor/codegen/memory_planning.py`](https://github.com/pytorch/pytorch/blob/2e9b4aff8d49b22bbebf288ccbf63983c51e45f0/torch/_inductor/codegen/memory_planning.py#L35-L132)

```python
class LiveRange:
    """
    A range where a given tensor is live.  Begin and end are both counters
    representing points in the program of grouped memory operations.
    Begin is inclusive, end is exclusive.

    Invariant: begin <= end
    """

    begin: float  # int | +/-inf
    end: float  # int | +/-inf

    def contains(self, other: LiveRange):
        """Is other entirely within self"""
        return self.begin <= other.begin and other.end <= self.end

    def join(self, other: LiveRange):
        """Combine two ranges using a union operation"""
        return LiveRange(min(self.begin, other.begin), max(self.end, other.end))

    def __len__(self):
        return self.end - self.begin


class LiveRanges:
    """
    A collection of LiveRange regions, allowing for non-contiguous
    live regions.

    Invariant: LiveRanges.ranges is in sorted order and non-overlapping
    """

    def __init__(self, ranges: Iterable[LiveRange]):
        ranges = [*sorted(ranges, key=lambda x: x.begin)]
        self.ranges = ranges[:1]
        for r in ranges[1:]:
            if self.ranges[-1].begin > r.begin:
                raise AssertionError("ranges must be sorted by begin")
            if self.ranges[-1].end >= r.begin:
                self.ranges[-1] = LiveRange.join(self.ranges[-1], r)
            else:
                self.ranges.append(r)

    def overlaps(self, other: LiveRanges):
        """Check if any pair of ranges in self and other overlap"""
        left = collections.deque(self.ranges)
        right = collections.deque(other.ranges)
        while left and right:
            if left[0].begin > right[0].begin:
                left, right = right, left
            if left[0].begin > right[0].begin:
                raise AssertionError("left should begin no later than right")
            if left[0].end > right[0].begin:
                return True
            left.popleft()
        return False

    @property
    def begin(self):
        return self.ranges[0].begin

    @property
    def end(self):
        return self.ranges[-1].end

    def __repr__(self):
        return f"{self.__class__.__name__}([{', '.join(map(repr, self.ranges))}])"


class AllocationTreeNode:
    """
    Abstract base class for nodes in allocation pool.
    """

    def allocate(self, block: Allocation, is_last: bool) -> bool:
        """
        Try to assign block to a memory location in this pool.  Return True if
        an assignment was made.
        """
        return False

    def get_live_ranges(self) -> LiveRanges:
        """Aggregate LiveRanges for all objects below this in tree"""
        raise NotImplementedError

    def get_size_hint(self) -> int:
        """Number of bytes used for example inputs"""
        raise NotImplementedError

    def get_symbolic_size(self) -> sympy.Expr:
        """Number of bytes needed at runtime"""
        raise NotImplementedError

    def finalize(self, pool, offset) -> AllocationTreeNode:
        """Called after all allocations have been made"""
        return self

    def is_empty(self):
```

## 逐行讲解 / What's happening

1. **第 35-56 行 / Lines 35-56**:
   - 中文: 单个生命期是半开区间 `[begin, end)`；半开写法能让一个 tensor 刚死、另一个马上出生时安全复用。
   - English: A single lifetime is a half-open interval `[begin, end)`, which lets one tensor die exactly when another is born.
1. **第 67-76 行 / Lines 67-76**:
   - 中文: 构造器先排序，再把相接或相交的区间合并，保持内部表示简洁。
   - English: The constructor sorts ranges and merges touching or overlapping spans to keep the representation compact.
1. **第 78-90 行 / Lines 78-90**:
   - 中文: 两个 deque 像拉链一样向前走，只要发现右侧开始早于左侧结束，就说明重叠。
   - English: Two deques advance like a zipper; if the right range starts before the left range ends, the lifetimes overlap.
1. **第 104-132 行 / Lines 104-132**:
   - 中文: 抽象节点把“尝试分配”和“聚合生命期”变成统一接口，后面的 temporal/spatial split 都能复用。
   - English: The abstract node gives later temporal and spatial split strategies a shared allocation interface.

## 类比 / The analogy

像会议室排期：两场会只要时间不重叠，就能用同一间房；如果中间断开，仍然要把多个时间段记在同一个日历上。

Think of room scheduling. Two meetings can use the same room only if their times do not overlap; non-contiguous bookings still belong on one calendar.

## 自己跑一遍 / Try it yourself

```python
def overlap(a, b):
    return a[0] < b[1] and b[0] < a[1]

blocks = {'tmp1': (0, 3), 'tmp2': (3, 5), 'tmp3': (2, 4)}
for x in blocks:
    for y in blocks:
        if x < y:
            print(x, y, overlap(blocks[x], blocks[y]))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```
tmp1 tmp2 False
tmp1 tmp3 True
tmp2 tmp3 True
```

`tmp1` 在 3 结束，`tmp2` 也在 3 开始，所以半开区间让它们不冲突。

`tmp1` ends at 3 and `tmp2` starts at 3, so half-open intervals make them compatible.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **XLA / TVM memory planning** / **Other compilers also reuse buffers after liveness analysis.**
- **Gradient checkpointing** / **Checkpointing trades stored live ranges for recomputation.**
- **Arena allocators** / **Arena allocators often need a lifetime model before reuse is safe.**

## 注意事项 / Caveats / when it breaks

- **别把 shape 只看成大小** / **Two buffers may have equal bytes but incompatible symbolic layout requirements.**
- **别复用输出** / **User-visible outputs usually need their own ownership.**
- **动态 shape 更难** / **Symbolic sizes make containment and offset placement more subtle.**

## 延伸阅读 / Further reading

- PyTorch Inductor `memory_planning.py` permalink above
