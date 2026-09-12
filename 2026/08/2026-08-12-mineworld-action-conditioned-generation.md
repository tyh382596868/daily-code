---
date: 2026-08-12
topic: infrastructure
source: trending
repo: microsoft/mineworld
file: inference.py
permalink: https://github.com/microsoft/mineworld/blob/main/inference.py#L64-L138
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, infrastructure, world-model]
---

# MineWorld inference：动作先变 token，再生成下一段画面 / MineWorld Inference: Tokenize Actions Before Generating the Next Frames

> **一句话 / In one line**: `lvm_generate()` 把 Minecraft 视频帧和动作 JSONL 都转成 token，然后让 transformer 在动作条件下生成未来图像 token。 / `lvm_generate()` converts Minecraft frames and action JSONL into tokens, then asks the transformer to generate future image tokens conditioned on actions.

## 为什么重要 / Why this matters

交互式世界模型的推理入口要同时处理两条流：已看到的画面和即将执行的动作。MineWorld 的代码把初始视频帧 token 化，把未来动作转成 action token，再调用两种生成路径：普通自回归或 image diagonal acceleration。这个结构很适合学习“可控视频生成”的工程接口。

An interactive world model inference entrypoint has to manage two streams: observed frames and planned actions. MineWorld tokenizes the seed video frames, converts future controls into action tokens, and then calls either naive autoregressive generation or image diagonal acceleration. It is a useful engineering pattern for controllable video generation.

## 代码 / The code

`microsoft/mineworld` — [`inference.py`](https://github.com/microsoft/mineworld/blob/main/inference.py#L64-L138)

```python
def lvm_generate(args, model, output_dir, demo_video):
    input_mp4_path = os.path.join(args.data_root, demo_video)
    input_action_path = os.path.join(args.data_root, demo_video.replace('mp4','jsonl'))
    output_mp4_path = str(output_dir / demo_video)
    output_action_path = output_mp4_path.replace('.mp4', '.jsonl')
    os.system(f"cp {input_action_path} {output_action_path}")
    if os.path.exists(output_mp4_path):
        print(f"output path {output_mp4_path} exist")
        return {}

    device = model.transformer.device
    action_list = []
    mcdataset = MCDataset()
    with open(input_action_path, 'r') as f:
        for line in f:
            line = eval(line.strip(), {"__builtins__": None}, safe_globals)
            line['camera'] = np.array(line['camera'])
            act_index = mcdataset.get_action_index_from_actiondict(line, action_vocab_offset=8192)
            action_list.append(act_index)

    cap = cv2.VideoCapture(input_mp4_path)
    frames = []
    for frame_idx in range(0, args.demo_num):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"Error in reading frame {frame_idx}")
            continue
        cv2.cvtColor(frame, code=cv2.COLOR_BGR2RGB, dst=frame)
        frame = np.asarray(np.clip(frame, 0, 255), dtype=np.uint8)
        frames.append(torch.from_numpy(frame))
    frames = torch.stack(frames, dim=0).to(device)
    frames = frames.permute(0, 3, 1, 2).float() / 255.0
    frames = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(frames)

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        img_index = model.tokenizer.tokenize_images(frames)
    img_index = rearrange(img_index, '(b t) h w -> b t (h w)', b=1)

    action_all = action_list[args.demo_num: args.demo_num + args.frames]
    action_all = torch.tensor(action_all).unsqueeze(1).to(device)
    image_input = rearrange(img_index, 'b t c -> b (t c)')
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
        if args.accelerate_algo == 'naive':
            outputs = model.transformer.naive_generate(
                input_ids=image_input, max_new_tokens=TOKEN_PER_PIX*args.frames,
                action_all=action_all, top_k=args.top_k, top_p=args.top_p)
        elif args.accelerate_algo == 'image_diagd':
            outputs = model.transformer.img_diagd_generate(
                input_ids=image_input, max_new_tokens=TOKEN_PER_PIX*args.frames,
                action_all=action_all, windowsize=args.window_size,
                top_k=args.top_k, top_p=args.top_p)

    all_generated_tokens = outputs.tolist()[0]
    token2video(all_generated_tokens, model.tokenizer, str(output_path / demo_video), args.fps, device)
    return {"token_num": len(all_generated_tokens)}
```

## 逐行讲解 / What's happening

1. **路径配对 / Path pairing**:
   - 中文: `.mp4` 和 `.jsonl` 共享同名 stem，推理时必须一起读，一个给视觉上下文，一个给动作条件。
   - English: The `.mp4` and `.jsonl` share a stem and must be read together: one provides visual context, the other action conditioning.
2. **动作 token / Action tokens**:
   - 中文: 每行动作字典被转成离散 action index，并加上 `8192` offset，避免和图像 token 撞词表区间。
   - English: Each action dictionary becomes a discrete action index with an `8192` offset so action tokens do not collide with image-token IDs.
3. **图像 token / Image tokens**:
   - 中文: 原始帧转 RGB、归一化到 `[-1, 1]`，再由 tokenizer 压成图像 token 网格。
   - English: Frames are converted to RGB, normalized to `[-1, 1]`, and compressed into image-token grids by the tokenizer.
4. **两种生成路径 / Two generation paths**:
   - 中文: `naive_generate` 逐 token 生成；`img_diagd_generate` 用窗口式对角生成加速同一件事。
   - English: `naive_generate` decodes token by token; `img_diagd_generate` accelerates the same task with a windowed diagonal image schedule.

## 类比 / The analogy

像给游戏录像续写：先把前几帧画面压成胶片编号，再把接下来按键记录压成遥控器编号，播放器按两份编号合成下一段视频。

It is like extending a gameplay recording: first compress the seed frames into film codes, then compress upcoming key presses into controller codes, and let the player synthesize the next clip from both.

## 自己跑一遍 / Try it yourself

```python
TOKEN_PER_PIX = 4
actions = ["forward", "jump", "left"]
seed_frames = [[10, 11, 12, 13]]
action_tokens = [8192 + i for i, _ in enumerate(actions)]
image_input = sum(seed_frames, [])
generated = []
for act in action_tokens:
    generated.extend([(act + i) % 10 for i in range(TOKEN_PER_PIX)])
print("input", image_input)
print("actions", action_tokens)
print("generated", generated)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
input [10, 11, 12, 13]
actions [8192, 8193, 8194]
generated [2, 3, 4, 5, 3, 4, 5, 6, 4, 5, 6, 7]
```

这个玩具例子保留了核心接口：历史图像 token 是 prompt，未来动作 token 是条件，输出是未来图像 token。

This toy keeps the core interface: historical image tokens are the prompt, future action tokens are conditioning, and output tokens represent future frames.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **World Action Models** / **World Action Models**: 机器人 WAM 也把观测帧和动作 chunk 拼成联合条件。
- **Game world models** / **Game world models**: Minecraft 类交互模型常把键鼠动作当作和图像并列的离散 token。

## 注意事项 / Caveats / when it breaks

- **`eval` 读取 JSONL 有风险** / **`eval` for JSONL is risky**: 源码禁用了 builtins，但工程上更稳的是 `json.loads` 加 schema 校验。
- **token 区间必须隔离** / **Token ranges must be isolated**: action offset 如果和图像 token 范围重叠，模型会混淆“画面”和“按键”。

## 延伸阅读 / Further reading

- [MineWorld `lvm_generate`](https://github.com/microsoft/mineworld/blob/main/inference.py#L64-L138)
