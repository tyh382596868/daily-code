---
date: 2026-07-19
topic: wam
source: wam
repo: facebookresearch/DiT
file: models.py
permalink: https://github.com/facebookresearch/DiT/blob/main/models.py
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, wam, dit, classifier-free-guidance, conditioning]
build_role: classifier-free-guidance advanced variant
---

# DiT LabelEmbedder：训练时随机丢条件，采样时做 CFG / DiT LabelEmbedder: Drop Conditions During Training, Use CFG at Sampling

> **一句话 / In one line**: DiT 用一个额外的 class token 表示“无条件”，训练时随机替换 label，为 classifier-free guidance 准备同一个模型的有条件和无条件路径。 / DiT reserves one extra class token for "unconditional", randomly replaces labels during training, and prepares one model for both conditional and unconditional CFG paths.

## 为什么重要 / Why this matters

WAM 里的 CFG 不一定是文本 prompt，也可能是动作、任务、相机轨迹或 robot state。核心技巧一样：模型要见过“没有条件”的输入，采样时才能比较有条件预测和无条件预测，再放大条件带来的方向。

In a WAM, CFG is not necessarily text prompt guidance; it can be action, task, camera trajectory, or robot state guidance. The core trick is the same: the model must see "no condition" examples during training, so sampling can compare conditional and unconditional predictions and amplify the conditional direction.

## 代码 / The code

`facebookresearch/DiT` — [`models.py`](https://github.com/facebookresearch/DiT/blob/main/models.py)

```python
class LabelEmbedder(nn.Module):
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings
```

## 逐行讲解 / What's happening

1. **多开一个 embedding / Allocate one extra embedding**: 中文: `num_classes + use_cfg_embedding` 给无条件 token 留位置。 English: `num_classes + use_cfg_embedding` reserves a slot for the unconditional token.
2. **无条件 id 放在末尾 / The unconditional id sits at the end**: 中文: `self.num_classes` 正好是新增 token 的 index。 English: `self.num_classes` is exactly the new token index.
3. **训练随机 drop / Training drops labels randomly**: 中文: `dropout_prob` 控制多少 batch 变成无条件。 English: `dropout_prob` controls how many rows become unconditional.
4. **采样可强制 drop / Sampling can force the drop mask**: 中文: `force_drop_ids` 让同一个 batch 能构造 cond/uncond 对。 English: `force_drop_ids` lets sampling build conditional and unconditional pairs.
5. **模型结构不变 / The model architecture stays the same**: 中文: 最终仍然只是查 embedding table。 English: the final operation is still an embedding lookup.

## 类比 / The analogy

像训练一个厨师同时学“按菜谱做”和“不看菜谱凭默认口味做”。试菜时比较两盘差异，就知道菜谱到底把味道推向了哪里。

It is like training a chef both with a recipe and without one. At tasting time, comparing the two plates reveals the direction that the recipe contributes.

## 在 nanoWAM 中的位置 / Where this lives in your nano-WAM

这是 `classifier-free-guidance` 的条件 dropout 层。上游是任务/action/text conditioning tokenizer，下游是 DiT block 的条件向量。nanoWAM 里可以先为“无动作条件”或“空任务条件”保留一个 learned embedding；生产版还要支持多条件分别 drop，例如 text drop 但 action 不 drop。

This is the condition-dropout layer for `classifier-free-guidance`. Upstream is the task/action/text conditioning tokenizer; downstream is the DiT block condition vector. In a nanoWAM, reserve a learned embedding for "no action condition" or "empty task"; a production system should support dropping different condition types separately, such as dropping text while keeping actions.

## 自己跑一遍 / Try it yourself

```python
labels = [2, 5, 1, 4]
num_classes = 10
force_drop = [0, 1, 0, 1]

def token_drop(labels, force_drop):
    return [num_classes if drop else label for label, drop in zip(labels, force_drop)]

cond = labels
uncond_mix = token_drop(labels, force_drop)
print(cond)
print(uncond_mix)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[2, 5, 1, 4]
[2, 10, 1, 10]
```

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Wan2.1 CFG sampler** / **Wan2.1 CFG sampler**: 正负 prompt 分别 forward，再按 guidance scale 混合。 / Positive and negative prompts are forwarded separately, then mixed by guidance scale.
- **Diffusers unconditional prompt** / **Diffusers unconditional prompt**: 文本 CFG 常用空 prompt 作为无条件分支。 / Text CFG often uses an empty prompt as the unconditional branch.

## 注意事项 / Caveats / when it breaks

- **dropout_prob 不能为 0 / `dropout_prob` cannot be zero for CFG**: 没见过无条件输入，采样时的 uncond 分支就不可靠。 / Without unconditional training examples, the uncond branch is unreliable.
- **多条件要分开设计 / Multiple conditions need separate masks**: text、action、state 可能不该一起 drop。 / Text, action, and state may not share one drop mask.
- **guidance scale 会放大错误 / Guidance scale amplifies mistakes**: 条件分支若偏，CFG 会把偏差也放大。 / If the conditional branch is biased, CFG amplifies the bias too.

## 延伸阅读 / Further reading

- [DiT models.py](https://github.com/facebookresearch/DiT/blob/main/models.py)
- [DiT repository](https://github.com/facebookresearch/DiT)
