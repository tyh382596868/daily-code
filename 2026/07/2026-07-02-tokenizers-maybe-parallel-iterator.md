---
date: 2026-07-02
topic: huggingface
source: huggingface
repo: huggingface/tokenizers
file: tokenizers/src/utils/parallelism.rs
permalink: https://github.com/huggingface/tokenizers/blob/main/tokenizers/src/utils/parallelism.rs#L15-L106
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, huggingface, tokenizers, rayon]
---

# tokenizers 并行开关：同一个 iterator 可串行也可并行 / tokenizers Parallelism Switch: One Iterator, Serial or Parallel

> **一句话 / In one line**: `MaybeParallelIterator` 把 Rayon 并行和普通迭代包装成同一个接口，由环境变量和进程内 override 决定走哪条路。 / `MaybeParallelIterator` wraps Rayon and serial iteration behind one interface, controlled by an environment variable plus a process-local override.

## 为什么重要 / Why this matters

Tokenizer 常被放进 dataloader、web worker 或 fork 后的子进程。默认并行能快，但在某些进程模型里会制造线程池问题。HF tokenizers 没有把并行判断散落在每个 encode 函数里，而是集中在一个 trait：调用方只说“也许并行”，底层再决定。

Tokenizers often run inside dataloaders, web workers, or forked child processes. Parallelism is fast by default, but can cause thread-pool trouble in some process models. HF tokenizers avoids scattering checks through every encode path by centralizing the decision in one trait: callers say "maybe parallel," and this layer decides.

## 代码 / The code

`huggingface/tokenizers` — [`tokenizers/src/utils/parallelism.rs`](https://github.com/huggingface/tokenizers/blob/main/tokenizers/src/utils/parallelism.rs#L15-L106)

```rust
pub const ENV_VARIABLE: &str = "TOKENIZERS_PARALLELISM";

static USED_PARALLELISM: AtomicBool = AtomicBool::new(false);
static PARALLELISM: AtomicU8 = AtomicU8::new(0);

pub fn is_parallelism_configured() -> bool {
    std::env::var(ENV_VARIABLE).is_ok() || get_override_parallelism().is_some()
}

fn get_override_parallelism() -> Option<bool> {
    match PARALLELISM.load(Ordering::SeqCst) {
        0 => None,
        1 => Some(false),
        2 => Some(true),
        _ => unreachable!(),
    }
}

fn get_env_parallelism() -> bool {
    match std::env::var(ENV_VARIABLE) {
        Ok(mut v) => {
            v.make_ascii_lowercase();
            !matches!(v.as_ref(), "" | "off" | "false" | "f" | "no" | "n" | "0")
        }
        Err(_) => true,
    }
}

pub fn get_parallelism() -> bool {
    if let Some(parallel) = get_override_parallelism() {
        parallel
    } else {
        get_env_parallelism()
    }
}

pub fn set_parallelism(val: bool) {
    PARALLELISM.store(if val { 2 } else { 1 }, Ordering::SeqCst);
}

pub trait MaybeParallelIterator<P, S>
where
    P: ParallelIterator,
    S: Iterator<Item = P::Item>,
{
    fn into_maybe_par_iter(self) -> CondIterator<P, S>;
    fn into_maybe_par_iter_cond(self, cond: bool) -> CondIterator<P, S>;
}

impl<P, S, I> MaybeParallelIterator<P, S> for I
where
    I: IntoParallelIterator<Iter = P, Item = P::Item> + IntoIterator<IntoIter = S, Item = S::Item>,
    P: ParallelIterator,
    S: Iterator<Item = P::Item>,
{
    fn into_maybe_par_iter(self) -> CondIterator<P, S> {
        let parallelism = get_parallelism();
        if parallelism {
            USED_PARALLELISM.store(true, Ordering::SeqCst);
        }
        CondIterator::new(self, parallelism)
    }

    fn into_maybe_par_iter_cond(self, cond: bool) -> CondIterator<P, S> {
        if cond {
            self.into_maybe_par_iter()
        } else {
            CondIterator::from_serial(self)
        }
    }
}
```

## 逐行讲解 / What's happening

1. **第 15-18 行 / Lines 15-18**: 中文: `AtomicU8` 用 0/1/2 表示未设置/强制关/强制开，避免全局可变 bool 的竞态。 / English: `AtomicU8` encodes unset/forced-off/forced-on as 0/1/2 and avoids racing on global mutable state.
2. **第 25-33 行 / Lines 25-33**: 中文: 进程内 override 优先级高于环境变量。 / English: The in-process override takes precedence over the environment variable.
3. **第 36-44 行 / Lines 36-44**: 中文: 多种 false 写法都被接受，没设置时默认开启并行。 / English: Several false spellings are accepted, and parallelism defaults on when unset.
4. **第 83-101 行 / Lines 83-101**: 中文: 真正的切换点只有 `CondIterator::new(self, parallelism)`。 / English: The actual switch is just `CondIterator::new(self, parallelism)`.

## 类比 / The analogy

像超市结账：平时开多条收银通道；如果店里在盘点或人流很小，就临时只开一条。顾客走同一个入口，调度由收银台决定。

It is like supermarket checkout: normally several lanes are open, but during inventory or light traffic the store can use one lane. Customers enter the same queueing interface; the checkout area decides the schedule.

## 自己跑一遍 / Try it yourself

```python
import os

def get_parallelism():
    v = os.environ.get("TOKENIZERS_PARALLELISM")
    if v is None:
        return True
    return v.lower() not in {"", "off", "false", "f", "no", "n", "0"}

for value in [None, "false", "yes"]:
    if value is None:
        os.environ.pop("TOKENIZERS_PARALLELISM", None)
    else:
        os.environ["TOKENIZERS_PARALLELISM"] = value
    print(value, "=>", get_parallelism())
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
None => True
false => False
yes => True
```

中文: Python 版本只复刻环境变量解析；Rust 源码还把这个布尔值接到 Rayon 的条件并行 iterator。

English: The Python version only mirrors environment parsing; the Rust source wires the boolean into Rayon's conditional parallel iterator.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **PyTorch dataloader workers / PyTorch dataloader workers**: 中文: 进程和线程池交互复杂，通常需要全局开关。 / English: Processes and thread pools interact in subtle ways, so global switches are common.
- **Rayon `CondIterator` / Rayon `CondIterator`**: 中文: 把“并行还是串行”变成 iterator 类型层面的选择。 / English: It turns "parallel or serial" into an iterator-level choice.

## 注意事项 / Caveats / when it breaks

- **默认开启 / Default-on behavior**: 中文: 没设置环境变量时是并行，这对 fork-heavy 程序可能不是你想要的默认。 / English: Unset means parallel, which may not be the desired default in fork-heavy programs.
- **全局状态 / Global state**: 中文: `set_parallelism` 影响整个进程，库代码不要偷偷调用。 / English: `set_parallelism` affects the whole process, so library code should not call it casually.

## 延伸阅读 / Further reading

- Source permalink above.
- Rayon `ParallelIterator` and `rayon_cond::CondIterator`.
