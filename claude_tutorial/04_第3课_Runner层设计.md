# 第3课：Runner 层设计

## 学习目标
- 掌握 Runner 完整继承体系和各子类的定位
- 理解组件加载的具体实现（编码器、DiT、VAE）及其与 Offload/量化的集成方式
- 掌握不同任务类型的编码器运行流程差异
- 理解 MultiModelStruct（MoE 双模型切换）的设计
- 具备实现新 Runner 的能力

---

## 0. 与前两课的关系

第1课已经介绍了 BaseRunner（接口契约）和 DefaultRunner（通用实现）的框架骨架，第2课介绍了 Pipeline 如何创建和调用 Runner。

**本课聚焦于**：具体 Runner 子类**如何实现**那些接口——它们加载了哪些组件、如何处理不同任务的输入、彼此之间有什么差异，以及一些高级模式（MoE 双模型、音频驱动等）。

---

## 1. Runner 完整继承体系

```
BaseRunner (base_runner.py)
    │  接口契约：定义 load_xxx / run_xxx / 生命周期钩子
    │
    └── DefaultRunner (default_runner.py)
            │  通用实现：run_pipeline / run_main / run_segment / 任务绑定
            │
            ├── WanRunner (wan/wan_runner.py)
            │   │  @RUNNER_REGISTER("wan2.1")
            │   │  编码器：T5 + CLIP + WanVAE
            │   │  调度器：WanScheduler / WanSchedulerCaching
            │   │
            │   ├── Wan22MoeRunner (wan/wan_runner.py)
            │   │       @RUNNER_REGISTER("wan2.2_moe")
            │   │       双模型结构：MultiModelStruct
            │   │       高噪声模型 + 低噪声模型
            │   │
            │   ├── Wan22DenseRunner (wan/wan_runner.py)
            │   │       @RUNNER_REGISTER("wan2.2")
            │   │       Wan2.2 Dense 模型，VAE 改用 Wan2_2_VAE
            │   │
            │   ├── WanVaceRunner (wan/wan_vace_runner.py)
            │   │   │   @RUNNER_REGISTER("wan2.1_vace")
            │   │   │   视频编辑任务，额外加载 VaceVideoProcessor
            │   │   │
            │   │   └── Wan22MoeVaceRunner
            │   │           @RUNNER_REGISTER("wan2.2_moe_vace")
            │   │
            │   ├── WanAudioRunner (wan/wan_audio_runner.py)
            │   │   │   @RUNNER_REGISTER("seko_talk")
            │   │   │   音频驱动视频，额外加载 AudioEncoder + AudioAdapter
            │   │   │   覆盖 run_main 实现流式音频分段推理
            │   │   │
            │   │   └── Wan22AudioRunner
            │   │           @RUNNER_REGISTER("wan2.2_audio")
            │   │
            │   ├── WanDistillRunner (wan/wan_distill_runner.py)
            │   │       @RUNNER_REGISTER("wan2.1_distill")
            │   │       蒸馏模型，CFG 默认关闭
            │   │
            │   ├── WanSFRunner (wan/wan_sf_runner.py)
            │   │       @RUNNER_REGISTER("wan2.1_sf")
            │   │       Self-Forcing 推理模式
            │   │
            │   └── WanAnimateRunner (wan/wan_animate_runner.py)
            │           @RUNNER_REGISTER("wan2.1_animate")
            │
            ├── HunyuanVideo15Runner (hunyuan_video/hunyuan_video_15_runner.py)
            │   │   @RUNNER_REGISTER("hunyuan_video_1.5")
            │   │   编码器：Qwen2.5-VL + ByT5 + SigLIP + HunyuanVideo15VAE
            │   │   调度器：HunyuanVideo15Scheduler
            │   │   支持 SR（超分）推理
            │   │
            │   └── HunyuanVideo15DistillRunner
            │           @RUNNER_REGISTER("hunyuan_video_1.5_distill")
            │
            ├── QwenImageRunner (qwen_image/qwen_image_runner.py)
            │       @RUNNER_REGISTER("qwen_image")
            │       图像生成/编辑
            │
            ├── LTX2Runner (ltx2/ltx2_runner.py)
            │       @RUNNER_REGISTER("ltx2")
            │       音视频生成，使用 Gemma 编码器
            │
            ├── ZImageRunner (z_image/z_image_runner.py)
            │       @RUNNER_REGISTER("z_image")
            │       文生图
            │
            ├── SeedVRRunner (seedvr/seedvr_runner.py)
            │       @RUNNER_REGISTER("seedvr")
            │
            └── WorldPlayRunner (worldplay/*.py)
                    @RUNNER_REGISTER("worldplay_*")
                    世界模型
```

**关键观察**：

1. **Wan 家族最庞大**：WanRunner 衍生出 7+ 个子类，覆盖 MoE、VACE（视频编辑）、Audio（音频驱动）、Distill（蒸馏）、SF（Self-Forcing）等场景
2. **继承深度最多 4 层**：如 `BaseRunner → DefaultRunner → WanRunner → WanAudioRunner → Wan22AudioRunner`
3. **每个叶子节点都有 `@RUNNER_REGISTER`**：通过注册名与 Pipeline 的 `model_cls` 对应

---

## 2. WanRunner 深度剖析

WanRunner 是最核心的 Runner 实现，其他 Runner 大多参考它的模式。

**核心文件**：`lightx2v/models/runners/wan/wan_runner.py`

### 2.1 组件加载全景

WanRunner 需要加载 4 类组件，每个都涉及 Offload 和量化的配置：

```
WanRunner.load_model()  ← 继承自 DefaultRunner
    │
    ├── load_transformer()  → WanModel (DiT)
    ├── load_text_encoder() → T5EncoderModel
    ├── load_image_encoder()→ CLIPModel
    └── load_vae()          → WanVAE (encoder + decoder)
```

### 2.2 load_transformer()：加载 DiT 模型

```python
# wan_runner.py:72-79
def load_transformer(self):
    wan_model_kwargs = {
        "model_path": self.config["model_path"],
        "config": self.config,
        "device": self.init_device
    }
    lora_configs = self.config.get("lora_configs")
    if not lora_configs:
        model = WanModel(**wan_model_kwargs)
    else:
        model = build_wan_model_with_lora(
            WanModel, self.config, wan_model_kwargs,
            lora_configs, model_type="wan2.1"
        )
    return model
```

**要点**：
- `self.init_device` 由 `set_init_device()` 决定——如果开启 CPU Offload，模型先加载到 CPU；否则直接加载到 GPU
- LoRA 支持两种模式：**合并模式**（直接合入权重）和**动态模式**（运行时在线应用）

**LoRA 的两种模式**：

```python
# wan_runner.py:35-60
def build_wan_model_with_lora(wan_module, config, model_kwargs, lora_configs, model_type="high_noise_model"):
    lora_dynamic_apply = config.get("lora_dynamic_apply", False)

    if lora_dynamic_apply:
        # 动态模式：LoRA 权重不合并，运行时在线计算
        # 优点：可以随时切换 LoRA，支持量化模型
        model_kwargs["lora_path"] = lora_path
        model_kwargs["lora_strength"] = lora_strength
        model = wan_module(**model_kwargs)
    else:
        # 合并模式：LoRA 权重直接合入主权重
        # 优点：推理速度快（无额外计算）
        # 限制：不支持量化模型，不支持 lazy_load
        assert not config.get("dit_quantized", False)
        assert not config.get("lazy_load", False)
        model = wan_module(**model_kwargs)
        lora_adapter = LoraAdapter(model)
        lora_adapter.apply_lora(lora_configs, model_type=model_type)
    return model
```

### 2.3 load_text_encoder()：加载 T5 文本编码器

```python
# wan_runner.py:119-157
def load_text_encoder(self):
    # 1. 确定设备（Offload 逻辑）
    t5_offload = self.config.get("t5_cpu_offload", self.config.get("cpu_offload"))
    if t5_offload:
        t5_device = torch.device("cpu")
    else:
        t5_device = torch.device(AI_DEVICE)

    # 2. 确定权重路径（量化 vs 原始）
    t5_quantized = self.config.get("t5_quantized", False)
    if t5_quantized:
        t5_quant_scheme = self.config.get("t5_quant_scheme", None)
        t5_model_name = f"models_t5_umt5-xxl-enc-{tmp_t5_quant_scheme}.pth"
        t5_quantized_ckpt = find_torch_model_path(self.config, "t5_quantized_ckpt", t5_model_name)
        t5_original_ckpt = None
    else:
        t5_model_name = "models_t5_umt5-xxl-enc-bf16.pth"
        t5_original_ckpt = find_torch_model_path(self.config, "t5_original_ckpt", t5_model_name)

    # 3. 创建编码器
    text_encoder = T5EncoderModel(
        text_len=self.config["text_len"],
        dtype=torch.bfloat16,
        device=t5_device,
        checkpoint_path=t5_original_ckpt,
        tokenizer_path=tokenizer_path,
        cpu_offload=t5_offload,
        t5_quantized=t5_quantized,
        t5_quantized_ckpt=t5_quantized_ckpt,
        quant_scheme=t5_quant_scheme,
        ...
    )
    text_encoders = [text_encoder]  # 返回列表（有些模型有多个文本编码器）
    return text_encoders
```

**组件加载的统一模式**：

每个组件加载都遵循相同的三步模式：

```
1. 读取 Offload 配置 → 决定加载到 CPU 还是 GPU
2. 读取量化配置 → 决定加载原始权重还是量化权重
3. 创建模型实例 → 传入设备、权重路径、量化方案
```

这个模式在所有 `load_xxx()` 方法中重复出现。下面用表格对比：

| 组件 | Offload 配置 key | 量化配置 key | 模型类 |
|------|-----------------|-------------|--------|
| T5 | `t5_cpu_offload` | `t5_quantized` / `t5_quant_scheme` | `T5EncoderModel` |
| CLIP | `clip_cpu_offload` → `clip_encoder_offload` | `clip_quantized` / `clip_quant_scheme` | `CLIPModel` |
| VAE | `vae_cpu_offload` | 无（VAE 一般不量化） | `WanVAE` |
| DiT | `cpu_offload`（全局） | `dit_quantized` / `dit_quant_scheme` | `WanModel` |

**为什么 Offload 配置 key 的命名不统一？**

因为不同模型使用不同的编码器（第2课讲过），Pipeline 层在 `enable_offload()` 中将统一的 `text_encoder_offload` 映射为模型特定的 key（如 `t5_cpu_offload`）。Runner 直接读取这些具体的 key，避免了再做一次 model_cls 判断。

### 2.4 load_image_encoder()：条件加载

```python
# wan_runner.py:81-117
def load_image_encoder(self):
    image_encoder = None
    # 只有 I2V 等需要图像输入的任务才加载
    if self.config["task"] in ["i2v", "flf2v", "animate", "s2v", "rs2v"] \
       and self.config.get("use_image_encoder", True):
        # ... Offload + 量化配置逻辑 ...
        image_encoder = CLIPModel(
            dtype=torch.float16,
            device=clip_device,
            checkpoint_path=clip_original_ckpt,
            clip_quantized=clip_quantized,
            ...
        )
    return image_encoder  # T2V 任务返回 None
```

**设计要点**：图像编码器只在需要时加载。T2V（文生视频）任务没有图像输入，不需要 CLIP 编码器，节省显存。

### 2.5 load_vae()：编解码器复用

```python
# wan_runner.py:214-220
def load_vae(self):
    vae_encoder = self.load_vae_encoder()
    if vae_encoder is None or self.config.get("use_tae", False):
        vae_decoder = self.load_vae_decoder()
    else:
        vae_decoder = vae_encoder  # 编解码器共享同一个实例
    return vae_encoder, vae_decoder
```

**为什么 encoder 和 decoder 可以共享？**

WanVAE 的编码和解码功能在同一个模型类中（有 `encode()` 和 `decode()` 两个方法）。当两者都需要时，不必创建两个实例，直接复用。

**但有两种例外**：
1. T2V 任务：不需要 encoder（没有输入图像需要编码），只需要 decoder
2. 使用 TAE（Tiny Auto Encoder）：decoder 用轻量版 VAE，encoder 还是完整版

```python
# wan_runner.py:166-187
def load_vae_encoder(self):
    # ...
    if self.config["task"] not in ["i2v", "flf2v", "animate", "vace", "s2v", "rs2v"]:
        return None  # T2V 不需要 encoder
    else:
        return self.vae_cls(**vae_config)

# wan_runner.py:189-212
def load_vae_decoder(self):
    # ...
    if self.config.get("use_tae", False):
        # 使用轻量版 VAE decoder（速度快但质量略低）
        vae_decoder = self.tiny_vae_cls(vae_path=tae_path, ...)
    else:
        vae_decoder = self.vae_cls(**vae_config)
    return vae_decoder
```

---

## 3. 编码器运行方法详解

第1课介绍过 `DefaultRunner.init_modules()` 通过方法绑定将不同任务路由到不同的编码方法：

```python
if self.config["task"] == "i2v":
    self.run_input_encoder = self._run_input_encoder_local_i2v
elif self.config["task"] == "t2v":
    self.run_input_encoder = self._run_input_encoder_local_t2v
# ...
```

下面深入看 WanRunner 的各个编码方法如何工作。

### 3.1 I2V 任务编码流程

I2V 需要处理：图像 + 文本 → latent 空间输入

```python
# default_runner.py:280-288
def _run_input_encoder_local_i2v(self):
    # 1. 读取图像
    img, img_ori = self.read_image_input(self.input_info.image_path)

    # 2. 图像编码（CLIP）→ 语义特征
    clip_encoder_out = self.run_image_encoder(img)

    # 3. VAE 编码 → latent 表示
    vae_encode_out, latent_shape = self.run_vae_encoder(
        img_ori if self.vae_encoder_need_img_original else img
    )
    self.input_info.latent_shape = latent_shape

    # 4. 文本编码（T5）→ 文本特征
    text_encoder_output = self.run_text_encoder(self.input_info)

    # 5. 清理显存
    torch_device_module.empty_cache()
    gc.collect()

    # 6. 组合所有编码器输出
    return self.get_encoder_output_i2v(
        clip_encoder_out, vae_encode_out, text_encoder_output, img
    )
```

**为什么图像要同时经过 CLIP 和 VAE 两个编码器？**

```
CLIP 编码器：提取图像的"语义信息"（高层特征）
             "这是一只猫在草地上" → 指导生成的内容方向

VAE 编码器：提取图像的"像素信息"（低层特征）
             精确的颜色、构图、纹理 → 保证第一帧的视觉一致性
```

两者的信息互补，缺一不可：
- 只有 CLIP：生成的视频内容相关但第一帧不精确
- 只有 VAE：第一帧精确但后续帧可能偏离语义

### 3.2 T2V 任务编码流程

T2V 只有文本输入，简单得多：

```python
# default_runner.py:291-299
def _run_input_encoder_local_t2v(self):
    # 1. 直接计算 latent shape（无需图像，用配置中的 target_height/width）
    self.input_info.latent_shape = self.get_latent_shape_with_target_hw()

    # 2. 文本编码
    text_encoder_output = self.run_text_encoder(self.input_info)

    # 3. 无图像编码输出
    return {
        "text_encoder_output": text_encoder_output,
        "image_encoder_output": None,
    }
```

### 3.3 WanRunner 的 run_text_encoder() 实现

```python
# wan_runner.py:243-281
def run_text_encoder(self, input_info):
    # lazy_load 模式：运行前加载，运行后卸载
    if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
        self.text_encoders = self.load_text_encoder()

    # 选择 prompt（是否使用增强版）
    prompt = input_info.prompt_enhanced if self.config["use_prompt_enhancer"] else input_info.prompt
    neg_prompt = input_info.negative_prompt

    # CFG 并行模式：正负样本分到不同 GPU
    if self.config.get("enable_cfg", False) and self.config["cfg_parallel"]:
        cfg_p_group = self.config["device_mesh"].get_group(mesh_dim="cfg_p")
        cfg_p_rank = dist.get_rank(cfg_p_group)
        if cfg_p_rank == 0:
            context = self.text_encoders[0].infer([prompt])
            # 补零对齐到 text_len
            context = torch.stack([
                torch.cat([u, u.new_zeros(self.config["text_len"] - u.size(0), u.size(1))])
                for u in context
            ])
            text_encoder_output = {"context": context}
        else:
            context_null = self.text_encoders[0].infer([neg_prompt])
            context_null = torch.stack([...])
            text_encoder_output = {"context_null": context_null}
    else:
        # 非并行模式：在同一个 GPU 上编码正负样本
        context = self.text_encoders[0].infer([prompt])
        context = torch.stack([...])  # 补零对齐
        if self.config.get("enable_cfg", False):
            context_null = self.text_encoders[0].infer([neg_prompt])
            context_null = torch.stack([...])
        else:
            context_null = None
        text_encoder_output = {
            "context": context,
            "context_null": context_null,
        }

    # lazy_load 模式：用完就卸载
    if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
        del self.text_encoders[0]
        torch_device_module.empty_cache()
        gc.collect()

    return text_encoder_output
```

**lazy_load 模式的设计模式**：

```
加载 → 使用 → 卸载 → 清理显存
```

这个模式在所有编码器运行方法中重复出现。它让显存占用变成"脉冲式"而非"常驻式"：

```
常驻模式（无 lazy_load）：
  T5 ████████████████████████████████████
  CLIP ██████████████████████████████████
  DiT ███████████████████████████████████
  VAE ███████████████████████████████████
  显存 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓

脉冲模式（lazy_load）：
  T5  ████
  CLIP     ███
  DiT          ██████████████████████
  VAE                                ████
  显存 ▓▓▓▓     ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
```

### 3.4 WanRunner 的 run_vae_encoder() 实现

这是最复杂的编码器方法之一，因为它涉及分辨率计算和分布式对齐：

```python
# wan_runner.py:360-407
def run_vae_encoder(self, first_frame, last_frame=None):
    if self.config.get("resize_mode", None) is None:
        h, w = first_frame.shape[2:]
        aspect_ratio = h / w
        max_area = self.config["target_height"] * self.config["target_width"]

        # 计算 latent 空间尺寸
        # latent_h = 原图高 ÷ vae_stride ÷ patch_size（向下取整再对齐）
        ori_latent_h = round(np.sqrt(max_area * aspect_ratio)
                             // self.config["vae_stride"][1]
                             // self.config["patch_size"][1]
                             * self.config["patch_size"][1])
        ori_latent_w = round(np.sqrt(max_area / aspect_ratio)
                             // self.config["vae_stride"][2]
                             // self.config["patch_size"][2]
                             * self.config["patch_size"][2])

        # 分布式场景下调整 latent 尺寸使其可被均匀切分
        if dist.is_initialized() and dist.get_world_size() > 1:
            latent_h, latent_w, world_size_h, world_size_w = \
                self._adjust_latent_for_grid_splitting(
                    ori_latent_h, ori_latent_w, dist.get_world_size()
                )
        else:
            latent_h, latent_w = ori_latent_h, ori_latent_w

        latent_shape = self.get_latent_shape_with_lat_hw(latent_h, latent_w)
    # ...
    vae_encoder_out = self.get_vae_encoder_output(
        first_frame, latent_h, latent_w, last_frame, ...
    )
    return vae_encoder_out, latent_shape
```

**vae_stride 的含义**：

`vae_stride` 是 VAE 编解码器在三个维度上的**下采样倍率**，格式为 `(temporal_stride, height_stride, width_stride)`。

```
vae_stride = (4, 8, 8) 的含义：
    时间维度：每 4 帧压缩为 1 个 latent 帧
    空间高度：每 8 个像素压缩为 1 个 latent 像素
    空间宽度：每 8 个像素压缩为 1 个 latent 像素
```

**形象理解**：VAE 就像一个"压缩器"，把高维的视频像素空间压缩到低维的 latent 空间。`vae_stride` 描述的就是这个压缩比。

```
原始视频（像素空间）                    latent 空间
┌──────────────────┐                  ┌────────┐
│                  │  VAE Encoder     │        │
│  81帧 × 480 × 832│  ──────────→    │ 21×60×104│
│  (像素)          │  vae_stride      │ (latent)│
│                  │  = (4, 8, 8)     │        │
└──────────────────┘                  └────────┘
    时间：81帧                         时间：21帧   (81-1)/4+1=21
    高度：480px                        高度：60     480/8=60
    宽度：832px                        宽度：104    832/8=104
```

**为什么时间维度的计算是 `(N-1)/4+1` 而不是 `N/4`？**

因为视频的第一帧需要特殊处理。VAE 在时间维度上的压缩策略是：**第一帧单独保留，剩下的帧每 4 帧压缩为 1 帧**。

```
原始帧：  [F1] [F2 F3 F4 F5] [F6 F7 F8 F9] ... [F78 F79 F80 F81]
           │       │              │                    │
           ▼       ▼              ▼                    ▼
latent帧：[L1]    [L2]           [L3]        ...     [L21]

第一帧单独 → 1 帧
剩余 80 帧 → 80/4 = 20 帧
总计 → 1 + 20 = 21 帧
公式 → (81-1)/4 + 1 = 21
```

这就是为什么 I2V 任务要求 `(num_frames - 1) % vae_stride[0] == 0`（第2课 `auto_calc_config` 中的帧数校正逻辑）。如果不满足这个条件，最后一组帧凑不满 4 帧，VAE 无法处理。

**不同模型的 vae_stride 对比**：

| 模型 | vae_stride | 压缩比 | 含义 |
|------|-----------|--------|------|
| Wan 2.1/2.2 | (4, 8, 8) | 4×8×8 = 256 | 每 256 个像素体素压缩为 1 个 latent |
| HunyuanVideo 1.5 | (4, 16, 16) | 4×16×16 = 1024 | 空间压缩更狠，latent 更小 |

HunyuanVideo 的空间下采样是 16 倍（而非 8 倍），意味着同样分辨率的视频，它的 latent 空间更小（60×104 → 30×52），DiT 计算量更少，但 VAE 的重建精度可能略低。

**vae_stride 在代码中的三个关键用途**：

```python
# 用途1：计算 latent shape（像素 → latent）
latent_h = target_height // vae_stride[1]      # 480 / 8 = 60
latent_w = target_width // vae_stride[2]        # 832 / 8 = 104
latent_t = (num_frames - 1) // vae_stride[0] + 1  # (81-1) / 4 + 1 = 21

# 用途2：从 latent 反算像素尺寸（latent → 像素）
pixel_h = latent_h * vae_stride[1]              # 60 * 8 = 480
pixel_w = latent_w * vae_stride[2]              # 104 * 8 = 832

# 用途3：帧数合法性校正（set_config.py:auto_calc_config）
if config["target_video_length"] % config["vae_stride"][0] != 1:
    # 向下对齐：确保 (num_frames - 1) 能被 vae_stride[0] 整除
    config["target_video_length"] = config["target_video_length"] // config["vae_stride"][0] * config["vae_stride"][0] + 1
```

---

**latent_shape 的含义**：

```python
# wan_runner.py:468-475
def get_latent_shape_with_lat_hw(self, latent_h, latent_w):
    latent_shape = [
        self.config.get("num_channels_latents", 16),  # C：latent 通道数
        (self.config["target_video_length"] - 1) // self.config["vae_stride"][0] + 1,  # T：时间维度
        latent_h,   # H：空间高度
        latent_w,   # W：空间宽度
    ]
    return latent_shape
```

以 480×832、81帧、vae_stride=(4,8,8) 为例：
- C = 16（每个 latent 位置有 16 个通道，类似 RGB 有 3 个通道）
- T = (81-1) / 4 + 1 = 21
- H = 480 / 8 = 60
- W = 832 / 8 = 104
- latent_shape = [16, 21, 60, 104]

**DiT 的全部工作就是在这个 [16, 21, 60, 104] 的 latent 空间上做去噪**——相比原始视频的 [3, 81, 480, 832]，数据量从 3×81×480×832≈97M 降到了 16×21×60×104≈2.1M，压缩了约 46 倍。这就是 VAE + latent diffusion 的核心价值。

### 3.5 get_vae_encoder_output()：构造 VAE 输入

```python
# wan_runner.py:409-456
def get_vae_encoder_output(self, first_frame, lat_h, lat_w, last_frame=None, ...):
    h = lat_h * self.config["vae_stride"][1]  # latent → pixel 尺寸
    w = lat_w * self.config["vae_stride"][2]

    # 构造 mask：标记哪些帧是"已知的"
    msk = torch.ones(1, self.config["target_video_length"], lat_h, lat_w, ...)
    if last_frame is not None:
        msk[:, 1:-1] = 0   # 首尾帧已知，中间待生成
    else:
        msk[:, 1:] = 0     # 只有首帧已知

    # 构造 VAE 输入：首帧 + 零帧（待生成的帧）
    vae_input = torch.concat([
        torch.nn.functional.interpolate(first_frame.cpu(), size=(h, w), mode="bicubic").transpose(0, 1),
        torch.zeros(3, self.config["target_video_length"] - 1, h, w),
    ], dim=1).to(AI_DEVICE)

    # VAE 编码
    vae_encoder_out = self.vae_encoder.encode(vae_input.unsqueeze(0).to(GET_DTYPE()))

    # 拼接 mask 和 latent
    vae_encoder_out = torch.concat([msk, vae_encoder_out]).to(GET_DTYPE())
    return vae_encoder_out
```

**mask 的作用**：告诉 DiT 模型哪些帧是已知的（首帧/尾帧），哪些是需要生成的。

```
I2V 的 mask：[1, 0, 0, 0, ..., 0]        ← 首帧已知
FLF2V 的 mask：[1, 0, 0, ..., 0, 1]      ← 首尾帧已知
T2V：无 mask（没有已知帧）
```

### 3.6 get_encoder_output_i2v()：组合最终输出

```python
# wan_runner.py:458-466
def get_encoder_output_i2v(self, clip_encoder_out, vae_encoder_out, text_encoder_output, img=None):
    image_encoder_output = {
        "clip_encoder_out": clip_encoder_out,     # CLIP 语义特征
        "vae_encoder_out": vae_encoder_out,       # VAE latent + mask
    }
    return {
        "text_encoder_output": text_encoder_output,   # T5 文本特征
        "image_encoder_output": image_encoder_output,  # 图像特征
    }
```

这个字典就是 `self.inputs`，后续传给 `model.infer(self.inputs)` 使用。

### 3.7 Prompt Enhancer 的执行位置

Prompt Enhancer（提示词增强）在 `run_pipeline()` 中执行，位于**编码器运行之前**：

```python
# default_runner.py:469-484
def run_pipeline(self, input_info):
    self.input_info = input_info

    # ① Prompt Enhancer 在这里执行（编码之前）
    if self.config["use_prompt_enhancer"]:
        self.input_info.prompt_enhanced = self.post_prompt_enhancer()

    # ② 然后才运行编码器（文本编码器会用到增强后的 prompt）
    self.inputs = self.run_input_encoder()

    # ③ 最后运行主推理流程
    gen_video_final = self.run_main()
    return gen_video_final
```

**为什么必须放在 `run_input_encoder()` 之前？**

因为文本编码器需要使用增强后的 prompt：

```python
# wan_runner.py:247
prompt = input_info.prompt_enhanced if self.config["use_prompt_enhancer"] else input_info.prompt
```

如果 prompt enhancer 放在编码之后，文本编码器拿到的就是原始的短 prompt，增强就白做了。

**完整时序**：

```
run_pipeline(input_info)
    │
    ├── ① post_prompt_enhancer()        ← 调用远程 LLM 服务扩写 prompt
    │       "a cat" → "A fluffy orange tabby cat sits gracefully..."
    │       结果存入 input_info.prompt_enhanced
    │
    ├── ② run_input_encoder()           ← 编码所有输入
    │       ├── run_text_encoder()      ← 用 prompt_enhanced 进行 T5 编码
    │       ├── run_image_encoder()     ← CLIP 编码（I2V）
    │       └── run_vae_encoder()       ← VAE 编码（I2V）
    │
    └── ③ run_main()                    ← 去噪 + VAE 解码
```

**`post_prompt_enhancer()` 的实现**：

```python
# default_runner.py:419-433
def post_prompt_enhancer(self):
    while True:
        for url in self.config["sub_servers"]["prompt_enhancer"]:
            # 1. 检查远程服务是否空闲
            response = requests.get(
                f"{url}/v1/local/prompt_enhancer/generate/service_status"
            ).json()
            if response["service_status"] == "idle":
                # 2. 发送原始 prompt，获取增强结果
                response = requests.post(
                    f"{url}/v1/local/prompt_enhancer/generate",
                    json={
                        "task_id": generate_task_id(),
                        "prompt": self.config["prompt"],
                    },
                )
                enhanced_prompt = response.json()["output"]
                return enhanced_prompt
        # 所有服务器都忙，继续轮询（while True）
```

注意这是一个**忙等待（busy-wait）轮询**：如果所有 enhancer 服务都在忙，它会不断重试直到有空闲服务。这在高并发场景下可能成为瓶颈。

**Prompt Enhancer 的启用条件**（回顾第1课）：

```python
# default_runner.py:60-66
# 只有同时满足以下条件才启用：
# 1. 任务类型是 T2V（I2V 有图像参考，prompt 不那么重要）
# 2. 配置中指定了 prompt_enhancer 的服务地址
# 3. 服务实际可用（check_sub_servers 检查通过）
if self.config["task"] == "t2v" and self.config.get("sub_servers", {}).get("prompt_enhancer"):
    self.has_prompt_enhancer = True
    if not self.check_sub_servers("prompt_enhancer"):
        self.has_prompt_enhancer = False
```

---

## 4. HunyuanVideo15Runner 与 WanRunner 的差异

### 4.1 编码器差异

| 维度 | WanRunner | HunyuanVideo15Runner |
|------|-----------|---------------------|
| **文本编码器** | T5 (1个) | Qwen2.5-VL + ByT5 (2个) |
| **图像编码器** | CLIP | SigLIP |
| **VAE** | WanVAE | HunyuanVideo15VAE |
| **调度器** | WanScheduler (UniPC) | HunyuanVideo15Scheduler (Euler) |
| **vae_stride** | (4, 8, 8) | (4, 16, 16) |

### 4.2 文本编码器：单编码器 vs 双编码器

**WanRunner：单编码器**
```python
# wan_runner.py:156-157
text_encoder = T5EncoderModel(...)
text_encoders = [text_encoder]  # 列表中只有 1 个
```

**HunyuanVideo15Runner：双编码器**
```python
# hunyuan_video_15_runner.py:72-103
def load_text_encoder(self):
    # 编码器1：Qwen2.5-VL（主编码器，提供语义理解）
    text_encoder = Qwen25VL_TextEncoder(
        dtype=torch.float16,
        device=qwen25vl_device,
        checkpoint_path=text_encoder_path,
        ...
    )

    # 编码器2：ByT5（辅助编码器，提供字节级文本特征）
    byt5 = ByT5TextEncoder(
        config=self.config,
        device=byt5_device,
        ...
    )

    text_encoders = [text_encoder, byt5]  # 列表中有 2 个
    return text_encoders
```

**为什么 HunyuanVideo 需要两个文本编码器？**

Qwen2.5-VL 是主编码器，提供高层语义理解；ByT5 是字节级编码器，对文字渲染（如生成视频中的文字、数字）特别有帮助。两者互补。

### 4.3 图像编码器：CLIP vs SigLIP

WanRunner 使用 CLIP 的输出是一个特征向量，直接注入 DiT 的 cross-attention。

HunyuanVideo15Runner 使用 SigLIP，输出的是带空间位置的特征图 + mask：

```python
# hunyuan_video_15_runner.py:488-500
def run_image_encoder(self, first_frame, last_frame=None):
    input_image_np = self.resize_and_center_crop(first_frame, ...)
    vision_states = self.image_encoder.encode_images(input_image_np).last_hidden_state
    image_encoder_output = self.image_encoder.infer(vision_states)  # 进一步投影
    image_encoder_mask = torch.ones((1, image_encoder_output.shape[1]), ...)
    return image_encoder_output, image_encoder_mask  # 返回两个值
```

### 4.4 T2V 编码的差异

WanRunner 的 T2V 直接返回 `image_encoder_output: None`。

HunyuanVideo15Runner 的 T2V 需要构造**全零的 SigLIP 占位符**：

```python
# hunyuan_video_15_runner.py:436-453
def _run_input_encoder_local_t2v(self):
    self.input_info.latent_shape = self.get_latent_shape_with_target_hw()
    text_encoder_output = self.run_text_encoder(self.input_info)

    # 全零占位：告诉模型"没有图像输入"
    siglip_output = torch.zeros(
        1, self.vision_num_semantic_tokens, self.config["hidden_size"],
        dtype=torch.bfloat16
    ).to(AI_DEVICE)
    siglip_mask = torch.zeros(
        1, self.vision_num_semantic_tokens,
        dtype=torch.bfloat16, device=torch.device(AI_DEVICE)
    )

    return {
        "text_encoder_output": text_encoder_output,
        "image_encoder_output": {
            "siglip_output": siglip_output,
            "siglip_mask": siglip_mask,    # 全零 mask = "忽略图像"
            "cond_latents": None,
        },
    }
```

**为什么不直接传 None？** 因为 HunyuanVideo 的 DiT 网络结构固定接受 siglip_output 输入（它需要计算 cross-attention），传全零 + 全零 mask 比传 None 后在网络内部加 if-else 更简洁。

### 4.5 分辨率处理的差异

**WanRunner**：从图像实际尺寸计算 latent 尺寸
```python
# wan_runner.py:361-368
h, w = first_frame.shape[2:]
aspect_ratio = h / w
max_area = self.config["target_height"] * self.config["target_width"]
ori_latent_h = round(np.sqrt(max_area * aspect_ratio) // vae_stride // patch_size * patch_size)
```

**HunyuanVideo15Runner**：从 aspect_ratio 配置和 bucket 系统计算
```python
# hunyuan_video_15_runner.py:118-146
def get_latent_shape_with_target_hw(self, origin_size=None):
    if origin_size is None:
        width, height = self.config["aspect_ratio"].split(":")  # 如 "16:9"
    target_size = self.config["transformer_model_name"].split("_")[0]  # 如 "720p"
    target_height, target_width = self.get_closest_resolution_given_original_size(
        (int(width), int(height)), target_size
    )
    # ...
```

HunyuanVideo 的 bucket 系统会根据目标分辨率（360p/480p/720p/1080p）生成一组候选分辨率，然后找到与原图宽高比最接近的那个。

### 4.6 SR（超分）支持

HunyuanVideo15Runner 独有的特性——内置超分推理：

```python
# hunyuan_video_15_runner.py:31-46
def __init__(self, config):
    # 检查是否有 SR 版本
    if "video_super_resolution" in config and "sr_version" in config["video_super_resolution"]:
        self.sr_version = config["video_super_resolution"]["sr_version"]
    else:
        self.sr_version = None

    # 如果有 SR，创建独立的配置和调度器
    if self.sr_version is not None:
        self.config_sr = copy.deepcopy(config)
        self.config_sr["sample_shift"] = config["video_super_resolution"]["flow_shift"]
        self.config_sr["sample_guide_scale"] = config["video_super_resolution"]["guidance_scale"]
        self.config_sr["infer_steps"] = config["video_super_resolution"]["num_inference_steps"]

# hunyuan_video_15_runner.py:428-433
def run_vae_decoder(self, latents):
    if self.sr_version:
        latents = self.run_sr(latents)  # 先超分再解码
    images = super().run_vae_decoder(latents)
    return images
```

**SR 的推理流程**：

```
DiT 去噪 → 得到低分辨率 latents
    │
    ▼
run_sr(latents)  ← 又是一轮完整的去噪循环
    │  但使用独立的 model_sr、scheduler_sr、config_sr
    │
    ▼
得到高分辨率 latents → VAE 解码 → 高清视频
```

---

## 5. MultiModelStruct：MoE 双模型切换

Wan 2.2 MoE 版本使用两个独立的 DiT 模型，根据去噪阶段动态切换。

### 5.1 为什么需要两个模型？

扩散模型的去噪过程可以分为两个阶段：
- **高噪声阶段**（早期步骤）：决定整体结构和构图
- **低噪声阶段**（后期步骤）：细化纹理和细节

Wan 2.2 为每个阶段训练了专门的专家模型，类似 MoE（Mixture of Experts），但不是在 layer 级别切换，而是在整个模型级别切换。

### 5.2 MultiModelStruct 实现

```python
# wan_runner.py:490-578
class MultiModelStruct:
    def __init__(self, model_list, config, boundary=0.875, num_train_timesteps=1000):
        self.model = model_list  # [high_noise_model, low_noise_model]
        self.boundary = boundary
        self.boundary_timestep = self.boundary * num_train_timesteps  # 875
        self.cur_model_index = -1

    def infer(self, inputs):
        self.get_current_model_index()  # 根据 timestep 选择模型
        self.model[self.cur_model_index].infer(inputs)

    def get_current_model_index(self):
        if self.scheduler.timesteps[self.scheduler.step_index] >= self.boundary_timestep:
            # 高噪声阶段：用 high_noise_model
            self.scheduler.sample_guide_scale = self.config["sample_guide_scale"][0]
            # 如果需要 Offload，把 low_noise_model 卸载，加载 high_noise_model
            if self.config.get("cpu_offload", False) and self.config.get("offload_granularity") == "model":
                if self.cur_model_index == 1:  # 上一步用的是 low_noise
                    self.offload_cpu(model_index=1)
                    self.to_cuda(model_index=0)
            self.cur_model_index = 0
        else:
            # 低噪声阶段：用 low_noise_model
            self.scheduler.sample_guide_scale = self.config["sample_guide_scale"][1]
            if self.config.get("cpu_offload", False) and self.config.get("offload_granularity") == "model":
                if self.cur_model_index == 0:
                    self.offload_cpu(model_index=0)
                    self.to_cuda(model_index=1)
            self.cur_model_index = 1
```

**MultiModelStruct 的设计巧妙之处**：

它对外暴露和普通单模型完全一样的接口（`infer()`、`set_scheduler()`），DefaultRunner 的 `run_segment()` 调用 `self.model.infer(self.inputs)` 时完全不需要知道底层是一个模型还是两个模型。这是**透明代理模式**。

```
DefaultRunner.run_segment()
    │
    ▼
self.model.infer(inputs)
    │
    ├── 如果 model 是 WanModel：直接推理
    │
    └── 如果 model 是 MultiModelStruct：
            1. get_current_model_index() → 选择模型
            2. self.model[idx].infer(inputs) → 选中的模型推理
```

### 5.3 MoE Runner 的 load_transformer()

```python
# wan_runner.py:603-632
@RUNNER_REGISTER("wan2.2_moe")
class Wan22MoeRunner(WanRunner):
    def load_transformer(self):
        if not self.config.get("lazy_load", False):
            high_noise_model = WanModel(
                model_path=self.high_noise_model_path, config=self.config, device=self.init_device,
                model_type="wan2.2_moe_high_noise"
            )
            low_noise_model = WanModel(
                model_path=self.low_noise_model_path, config=self.config, device=self.init_device,
                model_type="wan2.2_moe_low_noise"
            )
            return MultiModelStruct(
                [high_noise_model, low_noise_model],
                self.config,
                self.config["boundary"]  # 切换边界（默认 0.875）
            )
        else:
            # lazy_load 模式：先创建空壳，用到时再加载
            model_struct = MultiModelStruct([None, None], self.config, self.config["boundary"])
            model_struct.low_noise_model_path = self.low_noise_model_path
            model_struct.high_noise_model_path = self.high_noise_model_path
            model_struct.init_device = self.init_device
            return model_struct
```

---

## 6. 特殊 Runner 模式

### 6.1 WanAudioRunner：音频驱动的生命周期

WanAudioRunner 最大的特点是**覆盖了 `run_main()`** 和多个生命周期钩子：

```python
# wan_audio_runner.py:701-772
def run_main(self):
    try:
        self.va_controller = VAController(self)

        if self.va_controller.reader is None:
            # 固定音频分段：复用 DefaultRunner 的 run_main
            return super().run_main()

        # 流式音频：无限循环，逐段生成
        self.va_controller.start()
        self.init_run()
        self.video_segment_num = 1000000  # "无限"分段
        segment_idx = 0

        while True:
            # 从音频流获取下一段音频
            audio_array, valid_duration = self.va_controller.reader.get_audio_segment()

            # 推理当前分段
            self.init_run_segment(segment_idx, audio_array)
            latents = self.run_segment(segment_idx)
            self.gen_video = self.run_vae_decoder(latents)
            self.end_run_segment(segment_idx, valid_duration)

            segment_idx += 1
    finally:
        self.end_run()
        if self.va_controller is not None:
            self.va_controller.clear()
```

**音频 Runner 的核心变化**：

| 生命周期钩子 | DefaultRunner | WanAudioRunner |
|-------------|---------------|----------------|
| `get_video_segment_num()` | 返回 1 | 返回 `len(audio_segments)` |
| `init_run()` | 准备 scheduler | + 初始化 audio_adapter，准备 prev_video |
| `init_run_segment()` | 记录 segment_idx | + 编码当前段音频，准备 prev_latents |
| `end_run_segment()` | gen_video_final = gen_video | + 更新 prev_video（前一段的视频作为下一段的条件） |
| `run_main()` | 固定循环 | 支持流式无限循环 |

**prev_video 的滚动机制**：

```
Segment 0: 首帧图像 → 生成视频段0 → prev_video = 视频段0
Segment 1: prev_video 的末尾帧 → 生成视频段1 → prev_video = 视频段1
Segment 2: prev_video 的末尾帧 → 生成视频段2 → prev_video = 视频段2
...
```

每一段的 `init_run_segment()` 会调用 `prepare_prev_latents(self.prev_video)`，把前一段视频的最后几帧编码为 latent，作为当前段的条件输入，从而保证视频段之间的时间连续性。

### 6.2 WanVaceRunner：视频编辑

VACE（Video Anything Cut and Edit）Runner 的特殊之处在于 VAE 编码器的输入：

```python
# wan_vace_runner.py:99-133
def run_vae_encoder(self, frames, ref_images, masks):
    # VACE 的输入不是单张图，而是：完整视频帧 + 参考图像 + mask
    if masks is None:
        latents = [self.vae_encoder.encode(frame.unsqueeze(0)) for frame in frames]
    else:
        # mask 区域：inactive（保留区域）和 reactive（编辑区域）分别编码
        masks = [torch.where(m > 0.5, 1.0, 0.0) for m in masks]
        inactive = [i * (1 - m) + 0 * m for i, m in zip(frames, masks)]
        reactive = [i * m + 0 * (1 - m) for i, m in zip(frames, masks)]
        inactive = [self.vae_encoder.encode(inact.unsqueeze(0)) for inact in inactive]
        reactive = [self.vae_encoder.encode(react.unsqueeze(0)) for react in reactive]
        latents = [torch.cat((u, c), dim=0) for u, c in zip(inactive, reactive)]
    # ...
```

**编辑区域分离编码**的设计让模型知道哪些区域需要保留（inactive），哪些需要重新生成（reactive）。

### 6.3 Wan22DenseRunner：vae_encoder_need_img_original

```python
# wan_runner.py:668-676
@RUNNER_REGISTER("wan2.2")
class Wan22DenseRunner(WanRunner):
    def __init__(self, config):
        super().__init__(config)
        self.vae_encoder_need_img_original = True  # ← 关键差异
        self.vae_cls = Wan2_2_VAE
```

回顾 `_run_input_encoder_local_i2v()` 中的这行：

```python
vae_encode_out, latent_shape = self.run_vae_encoder(
    img_ori if self.vae_encoder_need_img_original else img
)
```

Wan 2.1 的 VAE 编码器接受 tensor 格式的图像（经过 normalize 的 `img`），而 Wan 2.2 的 VAE 编码器接受 PIL Image 格式的原图（`img_ori`），因为它内部有自己的预处理逻辑。

这是一个典型的**通过标志位而非 if-else 控制行为**的设计——子类只需要设置一个属性，父类的代码自动适配。

---

## 7. DefaultRunner 的 init_run() 详解

第1课简略介绍了 `init_run()`，这里深入分析完整实现：

```python
# default_runner.py:343-361
def init_run(self):
    self.gen_video_final = None
    self.get_video_segment_num()  # 默认 1，AudioRunner 覆盖

    # lazy_load 模式：此时才真正加载 DiT
    if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
        self.model = self.load_transformer()
        self.model.set_scheduler(self.scheduler)

    # 调度器初始化（设置初始噪声、timesteps 等）
    self.model.scheduler.prepare(
        seed=self.input_info.seed,
        latent_shape=self.input_info.latent_shape,
        image_encoder_output=self.inputs["image_encoder_output"]
    )

    # Wan2.2 I2V 特殊处理：prepare 之后释放 vae_encoder_out
    if self.config.get("model_cls") == "wan2.2" and self.config["task"] in ["i2v", "s2v", "rs2v"]:
        self.inputs["image_encoder_output"]["vae_encoder_out"] = None
```

**调用时序**：

```
run_pipeline()
  │
  ├── run_input_encoder()    ← 编码输入（T5/CLIP/VAE），得到 self.inputs
  │
  └── run_main()
        │
        ├── init_run()        ← 用 self.inputs 初始化 scheduler
        │     │
        │     └── scheduler.prepare()  ← 设置初始噪声 latents
        │
        ├── for segment in segments:
        │     ├── init_run_segment()
        │     ├── run_segment()       ← step_pre → infer → step_post
        │     ├── run_vae_decoder()
        │     └── end_run_segment()
        │
        ├── process_images_after_vae_decoder()
        └── end_run()
```

---

## 8. 组件初始化的完整时序

把所有组件的加载和使用时机画在一起：

```
Pipeline.create_generator()
    │
    └── _init_runner(config)
            │
            ├── Runner.__init__(config)
            │     ├── set_init_device()    → 决定 CPU 还是 GPU
            │     └── init_scheduler()     → 创建 Scheduler（不加载权重）
            │
            └── runner.init_modules()
                  │
                  ├── load_model()  ← 非 lazy_load 时在这里加载
                  │     ├── load_transformer()    → DiT（最大的模型）
                  │     ├── load_text_encoder()   → T5/Qwen2.5-VL
                  │     ├── load_image_encoder()  → CLIP/SigLIP（可能为 None）
                  │     └── load_vae()            → VAE encoder + decoder
                  │
                  ├── model.set_scheduler(scheduler)  → 绑定调度器
                  │
                  ├── run_input_encoder = 任务方法绑定
                  │
                  ├── config.lock()     → 锁定配置
                  │
                  └── model.compile()   → 可选：torch.compile

Pipeline.generate()
    │
    └── runner.run_pipeline(input_info)
          │
          ├── run_input_encoder()
          │     ├── [lazy_load] load_text_encoder() → 用 → 卸载
          │     ├── [lazy_load] load_image_encoder() → 用 → 卸载
          │     └── [lazy_load] load_vae_encoder() → 用 → 卸载
          │
          └── run_main()
                ├── [lazy_load] load_transformer() → 这才真正加载 DiT
                ├── scheduler.prepare()
                ├── run_segment()  × N
                ├── run_vae_decoder()
                │     └── [lazy_load] load_vae_decoder() → 用 → 卸载
                └── end_run()
                      └── [lazy_load] del self.model → 卸载 DiT
```

**两种模式对比**：

| 时机 | 常规模式 | lazy_load 模式 |
|------|---------|---------------|
| init_modules() | 加载所有组件 | 什么都不加载 |
| run_input_encoder() | 直接使用已加载的编码器 | 即时加载 → 使用 → 卸载 |
| run_main() | 直接使用已加载的 DiT | 即时加载 DiT |
| run_vae_decoder() | 直接使用已加载的 VAE | 即时加载 → 使用 → 卸载 |
| end_run() | 清理调度器状态 | + 卸载 DiT |

---

## 9. process_images_after_vae_decoder()：后处理

```python
# default_runner.py:435-467
def process_images_after_vae_decoder(self):
    # 1. 格式转换
    self.gen_video_final = wan_vae_to_comfy(self.gen_video_final)

    # 2. 可选：视频插帧
    if "video_frame_interpolation" in self.config:
        self.gen_video_final = self.vfi_model.interpolate_frames(
            self.gen_video_final,
            source_fps=self.config.get("fps", 16),
            target_fps=target_fps,
        )

    # 3. 输出：返回 tensor 或保存文件
    if self.input_info.return_result_tensor:
        return {"video": self.gen_video_final}
    elif self.input_info.save_result_path is not None:
        save_to_video(
            self.gen_video_final,
            self.input_info.save_result_path,
            fps=fps,
            method="ffmpeg"
        )
        return {"video": None}
```

**wan_vae_to_comfy 做了什么？** 它将 VAE 输出的 tensor 格式（`[1, C, T, H, W]`，值域 `[-1, 1]`）转换为 ComfyUI 兼容的格式（`[T, H, W, C]`，值域 `[0, 1]`，uint8）。

---

## 10. switch_lora()：运行时切换 LoRA

DefaultRunner 支持在不重载模型的情况下切换 LoRA 权重：

```python
# default_runner.py:486-529
def switch_lora(self, lora_path: str, strength: float = 1.0):
    if lora_path == "":
        # 移除 LoRA
        self.model._remove_lora()
    else:
        # 切换 LoRA
        self.model._update_lora(lora_path, strength)
    return True
```

Wan22MoeRunner 覆盖了这个方法，支持对高噪声和低噪声模型分别切换 LoRA：

```python
# wan_runner.py:634-665
def switch_lora(self, high_lora_path=None, high_lora_strength=1.0,
                      low_lora_path=None, low_lora_strength=1.0):
    if high_lora_path is not None:
        self.model.model[0]._update_lora(high_lora_path, high_lora_strength)
    if low_lora_path is not None:
        self.model.model[1]._update_lora(low_lora_path, low_lora_strength)
```

---

## 11. 实现新 Runner 的完整步骤

假设要适配一个名为 "MyModel" 的新模型，它使用 BERT 文本编码器、ViT 图像编码器、自己的 VAE 和 DDIM 调度器。

### 步骤 1：创建 Runner 文件

```python
# lightx2v/models/runners/my_model/my_runner.py

from lightx2v.models.runners.default_runner import DefaultRunner
from lightx2v.utils.registry_factory import RUNNER_REGISTER

@RUNNER_REGISTER("my_model")
class MyModelRunner(DefaultRunner):
    def __init__(self, config):
        super().__init__(config)
        # 子类特定初始化
```

### 步骤 2：实现 4 个 load 方法

```python
    def load_transformer(self):
        from my_model.dit import MyDiT
        return MyDiT(model_path=self.config["model_path"],
                     config=self.config,
                     device=self.init_device)

    def load_text_encoder(self):
        # 遵循 Offload + 量化 的三步模式
        offload = self.config.get("bert_cpu_offload", self.config.get("cpu_offload"))
        device = torch.device("cpu") if offload else torch.device(AI_DEVICE)
        encoder = BertEncoder(device=device, cpu_offload=offload, ...)
        return [encoder]  # 必须返回列表

    def load_image_encoder(self):
        if self.config["task"] not in ["i2v"]:
            return None  # 只有需要时才加载
        return ViTEncoder(...)

    def load_vae(self):
        encoder = MyVAE(...) if self.config["task"] in ["i2v"] else None
        decoder = MyVAE(...)
        return encoder, decoder
```

### 步骤 3：实现调度器初始化

```python
    def init_scheduler(self):
        from my_model.scheduler import DDIMScheduler
        self.scheduler = DDIMScheduler(self.config)
```

### 步骤 4：实现编码器运行方法

```python
    def run_text_encoder(self, input_info):
        prompt = input_info.prompt
        context = self.text_encoders[0].infer([prompt])
        return {"context": context, "context_null": ...}

    def run_image_encoder(self, img):
        return self.image_encoder.encode(img)

    def run_vae_encoder(self, img):
        latent = self.vae_encoder.encode(img)
        latent_shape = [...]
        return latent, latent_shape

    def get_encoder_output_i2v(self, clip_out, vae_out, text_out, img=None):
        return {
            "text_encoder_output": text_out,
            "image_encoder_output": {"clip": clip_out, "vae": vae_out},
        }
```

### 步骤 5：注册触发

```python
# lightx2v/pipeline.py 顶部添加
from lightx2v.models.runners.my_model.my_runner import MyModelRunner  # noqa: F401
```

### 步骤 6：Pipeline 适配

```python
# pipeline.py __init__ 中添加
elif self.model_cls in ["my_model"]:
    self.vae_stride = (4, 8, 8)  # 根据你的 VAE 设置

# pipeline.py enable_offload 中添加
elif self.model_cls in ["my_model"]:
    self.bert_cpu_offload = text_encoder_offload
```

### 核心原则

1. **只覆盖必须覆盖的方法**：DefaultRunner 的 `run_segment()`、`run_main()`、`run_pipeline()` 通常可以直接复用
2. **遵循三步加载模式**：Offload → 量化 → 创建
3. **返回格式与 DefaultRunner 一致**：`text_encoders` 返回列表，`load_vae()` 返回元组
4. **如果有新的生命周期需求才覆盖钩子**：比如 AudioRunner 需要在 `init_run_segment` 中编码音频

---

## 12. 关键源码文件索引

| 文件 | 行号 | 内容 |
|------|------|------|
| **BaseRunner** | | |
| `base_runner.py` | 10-19 | 初始化，vae_encoder_need_img_original |
| `base_runner.py` | 21-51 | 组件加载接口 |
| `base_runner.py` | 101-134 | 生命周期钩子 |
| **DefaultRunner** | | |
| `default_runner.py` | 55-68 | 初始化，prompt_enhancer 检查 |
| `default_runner.py` | 70-99 | init_modules，任务方法绑定 |
| `default_runner.py` | 101-105 | set_init_device |
| `default_runner.py` | 125-132 | load_model 统一入口 |
| `default_runner.py` | 175-208 | run_segment 去噪循环 |
| `default_runner.py` | 219-243 | end_run 资源清理 |
| `default_runner.py` | 279-341 | 各任务的 _run_input_encoder_local_xxx |
| `default_runner.py` | 343-361 | init_run 详细实现 |
| `default_runner.py` | 362-393 | run_main |
| `default_runner.py` | 395-417 | run_vae_decoder / run_vae_decoder_stream |
| `default_runner.py` | 435-467 | process_images_after_vae_decoder |
| `default_runner.py` | 469-484 | run_pipeline |
| `default_runner.py` | 486-529 | switch_lora |
| **WanRunner** | | |
| `wan_runner.py` | 63-79 | WanRunner 注册与 load_transformer |
| `wan_runner.py` | 81-117 | load_image_encoder（CLIP） |
| `wan_runner.py` | 119-157 | load_text_encoder（T5） |
| `wan_runner.py` | 166-220 | load_vae_encoder / load_vae_decoder / load_vae |
| `wan_runner.py` | 222-235 | init_scheduler（WanScheduler 选择） |
| `wan_runner.py` | 243-281 | run_text_encoder |
| `wan_runner.py` | 289-300 | run_image_encoder |
| `wan_runner.py` | 354-407 | run_vae_encoder（分辨率计算） |
| `wan_runner.py` | 409-456 | get_vae_encoder_output（mask 构造） |
| `wan_runner.py` | 458-466 | get_encoder_output_i2v |
| `wan_runner.py` | 468-487 | get_latent_shape_with_lat_hw / target_hw |
| `wan_runner.py` | 490-578 | MultiModelStruct（MoE） |
| `wan_runner.py` | 581-632 | Wan22MoeRunner |
| `wan_runner.py` | 668-708 | Wan22DenseRunner |
| **HunyuanVideo15Runner** | | |
| `hunyuan_video_15_runner.py` | 29-56 | 初始化，SR 配置 |
| `hunyuan_video_15_runner.py` | 58-70 | init_scheduler |
| `hunyuan_video_15_runner.py` | 72-103 | load_text_encoder（双编码器） |
| `hunyuan_video_15_runner.py` | 105-116 | load_transformer（含 SR 模型） |
| `hunyuan_video_15_runner.py` | 267-293 | run_text_encoder |
| `hunyuan_video_15_runner.py` | 295-309 | load_image_encoder（SigLIP） |
| `hunyuan_video_15_runner.py` | 396-426 | run_sr |
| `hunyuan_video_15_runner.py` | 428-433 | run_vae_decoder（集成 SR） |
| `hunyuan_video_15_runner.py` | 436-453 | _run_input_encoder_local_t2v（全零占位） |
| `hunyuan_video_15_runner.py` | 462-480 | _run_input_encoder_local_i2v |
| **WanAudioRunner** | | |
| `wan_audio_runner.py` | 276-328 | 初始化，音频输入读取 |
| `wan_audio_runner.py` | 504-556 | prepare_prev_latents（前一段视频条件化） |
| `wan_audio_runner.py` | 570-571 | get_video_segment_num = len(audio_segments) |
| `wan_audio_runner.py` | 573-585 | init_run（初始化 audio_adapter） |
| `wan_audio_runner.py` | 593-621 | init_run_segment（编码音频段） |
| `wan_audio_runner.py` | 629-665 | end_run_segment（更新 prev_video） |
| `wan_audio_runner.py` | 701-772 | run_main（流式 vs 固定分段） |
| **WanVaceRunner** | | |
| `wan_vace_runner.py` | 17-31 | 初始化，VaceVideoProcessor |
| `wan_vace_runner.py` | 99-133 | run_vae_encoder（inactive/reactive 分离编码） |

---

## 13. 思考题

1. **为什么 load_text_encoder() 返回列表而不是单个对象？**
   - 提示：考虑 HunyuanVideo 的双编码器

   可能有模型用多个编码器

   **批改**：方向正确 ✅，但可以更具体。

   **参考答案**：因为不同模型使用的文本编码器数量不同：
   - WanRunner：只有 T5，返回 `[text_encoder]`，列表长度 1
   - HunyuanVideo15Runner：有 Qwen2.5-VL + ByT5，返回 `[text_encoder, byt5]`，列表长度 2

   如果接口设计成返回单个对象，HunyuanVideo 就无法在同一个接口里返回两个编码器。用列表统一接口后，`run_text_encoder()` 可以通过 `self.text_encoders[0]`、`self.text_encoders[1]` 分别使用。

   这体现了一个设计原则：**接口按最复杂的使用场景设计，简单场景退化为列表长度为 1 的特例**。

2. **MultiModelStruct 的 offload_granularity="model" 是什么意思？它与 "block" 和 "phase" 有何区别？**
   - 提示：考虑 MoE 双模型切换时的显存管理

   model的意思是将整个dit模型offload到cpu上，block和phase则是以更细的粒度做offload。

   **批改**：核心理解正确 ✅，但对 "model" 粒度的理解不够精确 ❌。

   `offload_granularity="model"` 不是简单地把整个 DiT 放到 CPU，而是专门为 **MoE 双模型场景**设计的——在高噪声模型和低噪声模型之间做整体切换时的 offload。

   看 `MultiModelStruct.get_current_model_index()` 中的逻辑：

   ```python
   if self.config.get("offload_granularity") == "model":
       if self.cur_model_index == 1:  # 上一步用的是 low_noise
           self.offload_cpu(model_index=1)   # 把 low_noise 整个搬到 CPU
           self.to_cuda(model_index=0)       # 把 high_noise 整个搬到 GPU
   ```

   三种粒度对比：

   | 粒度 | 单位 | 适用场景 | 切换频率 |
   |------|------|---------|---------|
   | `"model"` | 整个 DiT 模型 | MoE 双模型切换 | 每次推理切换 1 次（高→低噪声） |
   | `"phase"` | 整个组件（T5/CLIP/DiT/VAE） | 单模型，组件间不同时使用 | 每次推理切换 3-4 次 |
   | `"block"` | DiT 的单个 Transformer Block | 单模型，逐 block 推理 | 每个 step 切换 N 次（N=block数） |

   `"model"` 粒度只在 Wan2.2 MoE 中有意义——它确保同一时刻 GPU 上只有一个完整的 DiT 模型，另一个在 CPU 上等待。对于非 MoE 的普通 Runner，这个选项没有意义。

3. **WanAudioRunner 为什么要覆盖 run_main() 而不是仅覆盖生命周期钩子？**
   - 提示：看流式音频输入的 while True 循环

   **参考答案**：因为 DefaultRunner 的 `run_main()` 的循环结构是**固定次数**的 `for segment_idx in range(self.video_segment_num)`，而 WanAudioRunner 的流式模式需要一个**无限循环** `while True`，逐段从音频流中获取数据。

   具体来说，有三个结构性差异无法仅通过覆盖钩子实现：

   1. **循环条件不同**：固定 `for` → 无限 `while True`，需要从 `va_controller.reader` 动态获取音频段
   2. **错误恢复**：流式模式需要 try-except 捕获 pause_signal 后继续运行（而不是终止），这个 try-except 包裹了整个段的处理过程
   3. **控制流中断**：流式模式有 `wait`、`switch_image`、`blank_to_voice` 等控制指令，需要在循环内做分支跳转

   ```python
   # DefaultRunner.run_main() 的循环：
   for segment_idx in range(self.video_segment_num):  # 固定次数
       self.init_run_segment(segment_idx)
       latents = self.run_segment(segment_idx)
       self.gen_video = self.run_vae_decoder(latents)
       self.end_run_segment(segment_idx)

   # WanAudioRunner.run_main() 的循环：
   while True:  # 无限
       control = self.va_controller.next_control()  # ← 钩子里做不了
       if control.action == "wait":
           continue                                  # ← 钩子里做不了
       audio_array = self.va_controller.reader.get_audio_segment()
       try:
           self.init_run_segment(segment_idx, audio_array)  # 多了参数
           latents = self.run_segment(segment_idx)
           self.end_run_segment(segment_idx, valid_duration)  # 多了参数
       except Exception as e:
           if "pause_signal" in str(e):
               continue  # 暂停后继续，而非终止     # ← 钩子里做不了
   ```

   总结：当子类需要的不只是在固定节点"加戏"，而是要**改变控制流本身**（循环条件、异常处理、分支跳转）时，就必须覆盖整个方法，生命周期钩子不够用。

4. **如果要在 WanRunner 中支持一种新的任务类型 "audio2video"（音频转视频，无图像输入），需要改哪些地方？**
   - 提示：参考 T2V 和 AudioRunner 的实现

   **参考答案**：

   **必须修改的**：

   - `input_info.py`：新建 `A2VInputInfo` dataclass，包含 `audio_path`、`prompt`、`seed` 等字段，在 `init_empty_input_info()` 中添加 `elif task == "a2v"` 分支
   - `default_runner.py init_modules()`：添加 `elif self.config["task"] == "a2v": self.run_input_encoder = self._run_input_encoder_local_a2v`
   - `pipeline.py generate()`：确保 `audio_path` 参数能传递到 input_info
   - 实现 `_run_input_encoder_local_a2v()`：参考 T2V（无图像输入）+ AudioRunner（有音频输入）

   ```python
   # 新方法：类似 T2V（无图像）但多了音频编码
   def _run_input_encoder_local_a2v(self):
       self.input_info.latent_shape = self.get_latent_shape_with_target_hw()
       text_encoder_output = self.run_text_encoder(self.input_info)

       # 音频部分：参考 WanAudioRunner
       audio_segments, expected_frames, _, audio_num = self.read_audio_input(
           self.input_info.audio_path
       )

       return {
           "text_encoder_output": text_encoder_output,
           "image_encoder_output": None,  # 无图像输入
           "audio_segments": audio_segments,
           "expected_frames": expected_frames,
       }
   ```

   **可能需要的**：
   - 如果 audio2video 也需要多段生成，需要覆盖 `get_video_segment_num()` 和相关的生命周期钩子（参考 WanAudioRunner 的做法）
   - 如果需要独立的音频编码器，需要在 `load_model()` 或单独的方法中加载


---

## 下一课预告

第4课将深入网络层（DiT）设计，分析：
- base_model.py 的三阶段推理架构：pre_infer / transformer_infer / post_infer
- 权重加载机制：weights/ 目录的组织方式
- infer/ 与 weights/ 的分离设计
- 如何支持不同的 DiT 变体
