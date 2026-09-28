---
date: 2026-09-28
topic: vla
source: vla
repo: huggingface/lerobot
file: src/lerobot/processor/tokenizer_processor.py
permalink: https://github.com/huggingface/lerobot/blob/e595b7902714ba51f91e47523f66f89c5181b649/src/lerobot/processor/tokenizer_processor.py#L56-L220
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, vla, language-tokenization]
build_role: action-tokenizer advanced variant: language-side tokenizer boundary
---

# LeRobot tokenizer processor：把任务文字变成 observation token / LeRobot Tokenizer Processor: Turn Task Text into Observation Tokens

> **一句话 / In one line**: VLA 的语言条件最好在 processor 边界就变成 tensor，而不是塞进模型 forward 里临时处理。 / A VLA should turn language conditions into tensors at the processor boundary instead of doing ad-hoc tokenization inside `forward`.

## 为什么重要 / Why this matters

这段不是动作 tokenizer 本体，而是 action-tokenizer 课程的相邻关键边界：语言任务、子任务和 token mask 要以稳定字段进入 observation。这样训练、推理和 replay buffer 都看到同一种 batch 合同。

This is not the action tokenizer itself; it is an adjacent boundary in the action-tokenizer curriculum. Task text, subtask text, and masks enter observations as stable fields, so training, inference, and replay buffers all share one batch contract.

## 代码 / The code

`huggingface/lerobot` — [`src/lerobot/processor/tokenizer_processor.py`](https://github.com/huggingface/lerobot/blob/e595b7902714ba51f91e47523f66f89c5181b649/src/lerobot/processor/tokenizer_processor.py#L56-L220)

```python
@dataclass
@ProcessorStepRegistry.register(name="tokenizer_processor")
class TokenizerProcessorStep(ObservationProcessorStep):
    """
    Processor step to tokenize a natural language task description.

    This step extracts a task string from the `complementary_data` of an `EnvTransition`,
    tokenizes it using a Hugging Face `transformers` tokenizer, and adds the resulting
    token IDs and attention mask to the `observation` dictionary.

    Requires the `transformers` library to be installed.

    Attributes:
        tokenizer_name: The name of a pretrained tokenizer from the Hugging Face Hub (e.g., "bert-base-uncased").
        tokenizer: A pre-initialized tokenizer object. If provided, `tokenizer_name` is ignored.
        max_length: The maximum length to pad or truncate sequences to.
        task_key: The key in `complementary_data` where the task string is stored.
        padding_side: The side to pad on ('left' or 'right').
        padding: The padding strategy ('max_length', 'longest', etc.).
        truncation: Whether to truncate sequences longer than `max_length`.
        input_tokenizer: The internal tokenizer instance, loaded during initialization.
    """

    tokenizer_name: str | None = None
    tokenizer: Any | None = None  # Use `Any` for compatibility without a hard dependency
    max_length: int = 512
    task_key: str = "task"
    padding_side: str = "right"
    padding: str = "max_length"
    truncation: bool = True

    # Internal tokenizer instance (not part of the config)
    input_tokenizer: Any = field(default=None, init=False, repr=False)

    def __post_init__(self):
        """
        Initializes the tokenizer after the dataclass is created.

        It checks for the availability of the `transformers` library and loads the tokenizer
        either from a provided object or by name from the Hugging Face Hub.

        Raises:
            ImportError: If the `transformers` library is not installed.
            ValueError: If neither `tokenizer` nor `tokenizer_name` is provided.
        """
        if not _transformers_available:
            raise ImportError(
                "The 'transformers' library is not installed. "
                "Please install it with `pip install 'lerobot[transformers-dep]'` to use TokenizerProcessorStep."
            )

        if self.tokenizer is not None:
            # Use provided tokenizer object directly
            self.input_tokenizer = self.tokenizer
        elif self.tokenizer_name is not None:
            if AutoTokenizer is None:
                raise ImportError("AutoTokenizer is not available")
            self.input_tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        else:
            raise ValueError(
                "Either 'tokenizer' or 'tokenizer_name' must be provided. "
                "Pass a tokenizer object directly or a tokenizer name to auto-load."
            )

    def get_task(self, transition: EnvTransition) -> list[str] | None:
        """
        Extracts the task description(s) from the transition's complementary data.

        Args:
            transition: The environment transition.

        Returns:
            A list of task strings, or None if the task key is not found or the value is None.
        """
        complementary_data = transition.get(TransitionKey.COMPLEMENTARY_DATA)
        if complementary_data is None:
            raise ValueError("Complementary data is None so no task can be extracted from it")

        task = complementary_data[self.task_key]
        if task is None:
            raise ValueError("Task extracted from Complementary data is None")

        # Standardize to a list of strings for the tokenizer
        if isinstance(task, str):
            return [task]
        elif isinstance(task, list | tuple) and all(isinstance(t, str) for t in task):
            return list(task)

        return None

    def get_subtask(self, transition: EnvTransition) -> list[str] | None:
        """
        Extracts the subtask from the transition's complementary data.

        Args:
            transition: The environment transition.

        Returns:
            A list of subtask strings, or None if the subtask key is not found or the value is None.
        """
        complementary_data = transition.get(TransitionKey.COMPLEMENTARY_DATA)
        if complementary_data is None:
            return None

        subtask = complementary_data.get("subtask")
        if subtask is None:
            return None

        # Standardize to a list of strings for the tokenizer
        if isinstance(subtask, str):
            return [subtask]
        elif isinstance(subtask, list) and all(isinstance(t, str) for t in subtask):
            return subtask

        return None

    def observation(self, observation: RobotObservation) -> RobotObservation:
        """
        Tokenizes the task description and adds it to the observation dictionary.

        This method retrieves the task, tokenizes it, moves the resulting tensors to the
        same device as other data in the transition, and updates the observation.

        Args:
            observation: The original observation dictionary.

        Returns:
            The updated observation dictionary including token IDs and an attention mask.
        """
        task = self.get_task(self.transition)
        if task is None:
            raise ValueError("Task cannot be None")

        # Tokenize the task (this will create CPU tensors)
        tokenized_prompt = self._tokenize_text(task)

        # Detect the device from existing tensors in the transition to ensure consistency
        target_device = self._detect_device(self.transition)

        # Move new tokenized tensors to the detected device
        if target_device is not None:
            tokenized_prompt = {
                k: v.to(target_device) if isinstance(v, torch.Tensor) else v
                for k, v in tokenized_prompt.items()
            }

        # Create a new observation dict to avoid modifying the original in place
        new_observation = dict(observation)

        # Add tokenized data to the observation
        new_observation[OBS_LANGUAGE_TOKENS] = tokenized_prompt["input_ids"]
        new_observation[OBS_LANGUAGE_ATTENTION_MASK] = tokenized_prompt["attention_mask"].to(dtype=torch.bool)

        # Tokenize subtask if available
        subtask = self.get_subtask(self.transition)
        if subtask is not None:
            tokenized_subtask = self._tokenize_text(subtask)

            # Move new tokenized tensors to the detected device
            if target_device is not None:
                tokenized_subtask = {
                    k: v.to(target_device) if isinstance(v, torch.Tensor) else v
                    for k, v in tokenized_subtask.items()
                }

```

## 逐行讲解 / What's happening

1. **第 56-88 行 / Lines 56-88**:
   - 中文: dataclass 字段把 tokenizer 来源、长度、padding 和 truncation 变成可配置 contract。
   - English: The dataclass fields turn tokenizer source, length, padding, and truncation into a configurable contract.
1. **第 101-118 行 / Lines 101-118**:
   - 中文: 初始化时要么使用传入 tokenizer，要么按名称加载，否则直接失败。
   - English: Initialization either uses an injected tokenizer or loads one by name; otherwise it fails loudly.
1. **第 120-144 行 / Lines 120-144**:
   - 中文: 任务字段统一成 `list[str]`，让单样本和 batch 样本走同一路径。
   - English: Task fields are normalized to `list[str]`, so single samples and batches use the same path.
1. **第 185-207 行 / Lines 185-207**:
   - 中文: tokenized tensor 被移动到 observation 已有设备，再写入标准 key。
   - English: Tokenized tensors move to the observation device before being stored under standard keys.

## 类比 / The analogy

像餐厅后厨的备料台：菜单文字不能每到炒锅前才临时切菜，应该先在备料台切成统一规格。

It is like a prep station in a kitchen. The order text should not be chopped at the stove; it should be prepared into a standard shape before cooking starts.

## 在 nanoVLA 中的位置 / Where this lives in your nano-VLA

在 nanoVLA 里，这属于输入 processor 层：原始 task/subtask 字符串进入这里，输出 `language_tokens` 和 `attention_mask`，再交给 VLM prefix builder。它依赖 vision/state/action 之外的 batch contract；省掉它会让训练脚本、推理服务和数据集各自处理文本，最后很难复现。

In nanoVLA, this belongs in the input processor layer: raw task/subtask strings enter here, and `language_tokens` plus `attention_mask` go to the VLM prefix builder. If you omit it, training scripts, inference services, and datasets each tokenize text differently, which quickly breaks reproducibility.

## 自己跑一遍 / Try it yourself

```python
def tokenize(tasks, max_len=5):
    vocab={'pick':1,'red':2,'cube':3,'place':4}
    out=[]
    for text in tasks:
        ids=[vocab.get(w,0) for w in text.split()][:max_len]
        ids += [0]*(max_len-len(ids))
        out.append(ids)
    return out
print(tokenize(['pick red cube','place cube']))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```
[[1, 2, 3, 0, 0], [4, 3, 0, 0, 0]]
```

重点不是 tokenizer 多聪明，而是下游永远收到同形状、同 key 的 tensor。

The point is not tokenizer sophistication; it is that downstream code always receives tensors with the same shape and keys.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **OpenVLA action tokenizer** / **OpenVLA discretizes continuous actions behind a similarly explicit boundary.**
- **SmolVLA processor stack** / **SmolVLA uses processors to keep image, language, and state contracts stable.**
- **openpi transforms** / **openpi also separates data transforms from policy execution.**

## 注意事项 / Caveats / when it breaks

- **字段名是 API** / **Changing keys breaks saved datasets and processors.**
- **padding side 会影响 decoder** / **Left vs right padding matters for causal models.**
- **tokenizer 不是动作空间** / **Language tokens condition action prediction; action tokens or heads still need their own component.**

## 延伸阅读 / Further reading

- LeRobot processor source permalink above
