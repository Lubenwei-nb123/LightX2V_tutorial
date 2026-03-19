# 第2课：Pipeline 层设计

## 学习目标
- 理解 LightX2VPipeline 的职责边界
- 掌握 model_cls 到 Runner 的映射机制
- 理解各种 enable_xxx() 方法的设计
- 掌握配置如何在各层之间传递

---

## 1. Pipeline 层的职责

Pipeline 是用户与框架交互的唯一入口，它的职责是：

```
┌─────────────────────────────────────────────────────────────┐
│                    LightX2VPipeline                         │
│                                                             │
│  1. 用户接口：提供简洁的 API                                 │
│  2. 配置收集：收集并整合各种配置                             │
│  3. Runner 路由：根据 model_cls 选择对应的 Runner            │
│  4. 推理调度：调用 Runner 执行推理                          │
└─────────────────────────────────────────────────────────────┘
```

**核心原则**：Pipeline 只做"调度"，不做"执行"。具体的模型加载、推理逻辑都在 Runner 中。

---

## 2. Pipeline 类结构

**核心文件**：`lightx2v/pipeline.py`

### 2.1 @dict_like 装饰器

```python
# pipeline.py:36-59
def dict_like(cls):
    cls.__getitem__ = lambda self, key: getattr(self, key)
    cls.__setitem__ = lambda self, key, value: setattr(self, key, value)
    cls.__delitem__ = lambda self, key: delattr(self, key)
    cls.__contains__ = lambda self, key: hasattr(self, key)

    def update(self, *args, **kwargs):
        for arg in args:
            if isinstance(arg, dict):
                items = arg.items()
            else:
                items = arg
            for k, v in items:
                setattr(self, k, v)

    def get(self, key, default=None):
        return getattr(self, key, default)

    cls.get = get
    cls.update = update
    return cls
```

**作用**：让 Pipeline 实例可以像字典一样访问属性。

```python
pipe = LightX2VPipeline(...)

# 以下两种写法等价
pipe.model_path        # 属性访问
pipe["model_path"]     # 字典访问

# 以下两种写法等价
pipe.infer_steps = 50
pipe["infer_steps"] = 50

# 支持 update
pipe.update({"height": 480, "width": 832})

# 支持 get
pipe.get("height", 480)  # 如果没有则返回默认值
```

**为什么需要这个？**

因为后续 `set_config()` 函数需要把 Pipeline 的属性转换成配置字典，`dict_like` 让这个转换更方便。

---

### 2.2 __init__：初始化

```python
# pipeline.py:62-128
@dict_like
class LightX2VPipeline:
    def __init__(
        self,
        task,                    # 任务类型：t2v, i2v, t2i, i2i, flf2v, vace, ...
        model_path,              # 模型路径
        model_cls,               # 模型类型：wan2.1, hunyuan_video_1.5, qwen_image, ...
        sf_model_path=None,      # Self-Forcing 模型路径（可选）
        dit_original_ckpt=None,  # DiT 原始权重路径（可选）
        transformer_model_name=None,  # Transformer 子目录名（HunyuanVideo 用）
        ...
    ):
```

**初始化做了什么？**

```python
# 1. 保存基本参数
self.task = task
self.model_path = model_path
self.model_cls = model_cls

# 2. 根据 model_cls 设置模型特定的默认参数
if self.model_cls in ["wan2.1", "wan2.1_distill", ...]:
    self.vae_stride = (4, 8, 8)
    if self.model_cls.startswith("wan2.2"):
        self.use_image_encoder = False
elif self.model_cls in ["hunyuan_video_1.5", ...]:
    self.vae_stride = (4, 16, 16)
    self.num_channels_latents = 32
elif self.model_cls in ["ltx2"]:
    self.num_channels_latents = 128
    self.audio_mel_bins = 16

# 3. 处理 model_cls 别名
if model_cls in ["qwen-image", "qwen-image-2512", ...]:
    self.model_cls = "qwen_image"  # 统一为内部名称
    # 设置任务特定的 prompt 模板
    if self.task in ["i2i"]:
        self.prompt_template_encode = "..."
    elif self.task in ["t2i"]:
        self.prompt_template_encode = "..."

# 4. 初始化输入信息结构
self.input_info = init_empty_input_info(self.task)
```

**设计要点**：
- `model_cls` 决定了模型的基本配置（vae_stride、num_channels_latents 等）
- 支持 model_cls 别名（如 `qwen-image` → `qwen_image`）
- 不同任务类型有不同的 prompt 模板

---

### 2.3 支持的 model_cls 和 task

**model_cls（模型类型）**：

| model_cls | 说明 |
|-----------|------|
| `wan2.1` | Wan 2.1 基础模型 |
| `wan2.1_distill` | Wan 2.1 蒸馏模型 |
| `wan2.2_moe` | Wan 2.2 MoE 模型 |
| `wan2.2_moe_audio` | Wan 2.2 音频驱动 |
| `hunyuan_video_1.5` | HunyuanVideo 1.5 |
| `hunyuan_video_1.5_distill` | HunyuanVideo 1.5 蒸馏 |
| `qwen_image` | Qwen 图像生成/编辑 |
| `ltx2` | LTX-2 音视频生成 |
| `z_image` | Z-Image 文生图 |
| ... | ... |

**task（任务类型）**：

| task | 说明 | 输入 |
|------|------|------|
| `t2v` | 文生视频 | prompt |
| `i2v` | 图生视频 | prompt + image |
| `t2i` | 文生图 | prompt |
| `i2i` | 图像编辑 | prompt + image |
| `flf2v` | 首尾帧生成视频 | prompt + first_frame + last_frame |
| `vace` | 视频编辑 | prompt + video + mask |
| `s2v` | 主体驱动视频 | prompt + subject_image |
| `t2av` | 文生音视频 | prompt |
| `i2av` | 图生音视频 | prompt + image |
| `sr` | 视频超分 | video |

---

## 3. enable_xxx() 方法：配置收集器

Pipeline 提供了一系列 `enable_xxx()` 方法，用于收集各种优化配置。

### 3.1 设计模式：Builder Pattern

```python
# 用户代码
pipe = LightX2VPipeline(model_cls="wan2.1", task="i2v", model_path="...")

pipe.enable_offload(cpu_offload=True, offload_granularity="block")
pipe.enable_quantize(dit_quantized=True, quant_scheme="fp8-sgl")
pipe.enable_cache(cache_method="Tea", teacache_thresh=0.15)
pipe.enable_lora(lora_configs=[{"path": "...", "strength": 1.0}])

pipe.create_generator(infer_steps=40, height=480, width=832)
pipe.generate(prompt="...", image_path="...", save_result_path="...")
```

**这是 Builder 模式**：通过链式调用逐步构建配置，最后调用 `create_generator()` 完成构建。

### 3.2 enable_offload()：显存优化配置

```python
# pipeline.py:299-336
def enable_offload(
    self,
    cpu_offload=False,           # 是否启用 CPU Offload
    offload_granularity="block", # Offload 粒度：block / phase
    text_encoder_offload=False,  # 文本编码器 Offload
    image_encoder_offload=False, # 图像编码器 Offload
    vae_offload=False,           # VAE Offload
):
    self.cpu_offload = cpu_offload
    self.offload_granularity = offload_granularity
    self.vae_cpu_offload = vae_offload

    # 根据 model_cls 设置对应的编码器 Offload 配置
    if self.model_cls in ["wan2.1", "wan2.2_moe", ...]:
        self.t5_cpu_offload = text_encoder_offload      # Wan 用 T5
        self.clip_encoder_offload = image_encoder_offload  # Wan 用 CLIP

    elif self.model_cls in ["hunyuan_video_1.5", ...]:
        self.qwen25vl_cpu_offload = text_encoder_offload   # HunyuanVideo 用 Qwen2.5-VL
        self.siglip_cpu_offload = image_encoder_offload    # HunyuanVideo 用 SigLIP

    elif self.model_cls in ["qwen_image", ...]:
        self.qwen25vl_cpu_offload = text_encoder_offload

    elif self.model_cls == "ltx2":
        self.gemma_cpu_offload = text_encoder_offload      # LTX2 用 Gemma
```

**设计要点**：
- 统一的用户接口（`text_encoder_offload`）
- 内部根据 model_cls 映射到具体的编码器配置（`t5_cpu_offload` / `qwen25vl_cpu_offload`）
- 用户不需要知道每个模型用的是什么编码器

### 3.3 enable_quantize()：量化配置

```python
# pipeline.py:261-297
def enable_quantize(
    self,
    dit_quantized=False,              # DiT 是否量化
    text_encoder_quantized=False,     # 文本编码器是否量化
    image_encoder_quantized=False,    # 图像编码器是否量化
    dit_quantized_ckpt=None,          # DiT 量化权重路径
    quant_scheme="fp8-sgl",           # 量化方案
    skip_fp8_block_index=[0, 43, 44, 45, 46, 47],  # 跳过量化的 block
    ...
):
    self.dit_quantized = dit_quantized
    self.dit_quant_scheme = quant_scheme
    self.dit_quantized_ckpt = dit_quantized_ckpt

    # 根据 model_cls 设置对应的编码器量化配置
    if self.model_cls.startswith("wan"):
        self.t5_quant_scheme = quant_scheme
        self.t5_quantized = text_encoder_quantized
        self.clip_quant_scheme = quant_scheme
        self.clip_quantized = image_encoder_quantized

    elif self.model_cls in ["hunyuan_video_1.5", ...]:
        self.qwen25vl_quantized = text_encoder_quantized
        self.qwen25vl_quant_scheme = text_encoder_quant_scheme
```

**量化方案**：
- `fp8-sgl`：FP8 量化（SGL 格式）
- `fp8-torchao`：FP8 量化（TorchAO）
- `nvfp4`：NVIDIA FP4 量化

### 3.4 enable_cache()：缓存加速配置

```python
# pipeline.py:369-391
def enable_cache(
    self,
    cache_method="Tea",           # 缓存方法：Tea / Mag / TaylorSeer
    coefficients=[],              # TeaCache 系数
    teacache_thresh=0.15,         # TeaCache 阈值
    use_ret_steps=False,          # 是否使用 retention steps
    magcache_calibration=False,   # MagCache 校准模式
    magcache_K=6,                 # MagCache K 值
    magcache_thresh=0.24,         # MagCache 阈值
    magcache_retention_ratio=0.2, # MagCache 保留比例
    magcache_ratios=[],           # MagCache 各层比例
):
    self.feature_caching = cache_method

    if cache_method == "Tea":
        self.coefficients = coefficients
        self.teacache_thresh = teacache_thresh
        self.use_ret_steps = use_ret_steps

    elif cache_method == "Mag":
        self.magcache_calibration = magcache_calibration
        self.magcache_K = magcache_K
        self.magcache_thresh = magcache_thresh
        self.magcache_retention_ratio = magcache_retention_ratio
        self.magcache_ratios = magcache_ratios
```

### 3.5 enable_parallel()：并行配置

```python
# pipeline.py:393-398
def enable_parallel(self, cfg_p_size=1, seq_p_size=1, seq_p_attn_type="ulysses"):
    self.parallel = {
        "cfg_p_size": cfg_p_size,      # CFG 并行度
        "seq_p_size": seq_p_size,      # 序列并行度
        "seq_p_attn_type": seq_p_attn_type,  # 序列并行注意力类型
    }
```

**并行策略**：
- `cfg_p_size`：CFG 并行，正负样本分到不同 GPU
- `seq_p_size`：序列并行，长序列切分到不同 GPU
- `seq_p_attn_type`：`ulysses`（Ulysses 并行）或 `ring`（Ring Attention）

### 3.6 其他 enable 方法

```python
# enable_lightvae：轻量 VAE
def enable_lightvae(self, use_lightvae=False, use_tae=False, vae_path=None, tae_path=None):
    ...

# enable_lora：LoRA 配置
def enable_lora(self, lora_configs, lora_dynamic_apply=False):
    self.lora_configs = lora_configs
    self.lora_dynamic_apply = lora_dynamic_apply

# enable_compile：torch.compile 配置
def enable_compile(self):
    self.compile = True
    self.compile_shapes = [[480, 832], [720, 1280], ...]
```

---

## 4. create_generator()：配置整合与 Runner 创建

这是 Pipeline 最核心的方法，负责整合所有配置并创建 Runner。

### 4.1 方法签名

```python
# pipeline.py:130-188
def create_generator(
    self,
    attn_mode="flash_attn2",      # 注意力算子
    infer_steps=50,               # 推理步数
    num_frames=81,                # 视频帧数
    height=480,                   # 高度
    width=832,                    # 宽度
    guidance_scale=5.0,           # CFG scale
    sample_shift=5.0,             # 采样偏移
    fps=16,                       # 帧率
    config_json=None,             # JSON 配置文件（可选）
    ...
):
```

### 4.2 执行流程

```python
def create_generator(self, ...):
    # 1. 设置推理配置
    if config_json is not None:
        self.set_infer_config_json(config_json)  # 从 JSON 文件加载
    else:
        self.set_infer_config(...)  # 从参数设置

    # 2. 整合所有配置
    config = set_config(self)  # 把 Pipeline 属性转换为 config 字典
    validate_config_paths(config)  # 验证路径是否存在

    # 3. 初始化并行环境（如果需要）
    if config["parallel"]:
        platform_device = PLATFORM_DEVICE_REGISTER.get(os.getenv("PLATFORM", "cuda"), None)
        platform_device.init_parallel_env()
        set_parallel_config(config)

    # 4. 创建 Runner
    self.runner = self._init_runner(config)

    logger.info(f"Initializing {self.model_cls} runner for {self.task} task...")
    logger.info("LightGenerator initialized successfully!")
```

### 4.3 set_infer_config()：设置推理参数

```python
# pipeline.py:190-238
def set_infer_config(self, attn_mode, rope_type, infer_steps, ...):
    # 基本参数
    self.infer_steps = infer_steps
    self.target_width = width
    self.target_height = height
    self.target_video_length = num_frames
    self.sample_guide_scale = guidance_scale
    self.sample_shift = sample_shift
    self.fps = fps

    # 根据 guidance_scale 决定是否启用 CFG
    if self.sample_guide_scale == 1:
        self.enable_cfg = False
    else:
        self.enable_cfg = True

    # 根据 model_cls 设置注意力类型
    if self.model_cls.startswith("wan"):
        # Wan 模型有三种注意力
        self.self_attn_1_type = attn_mode
        self.cross_attn_1_type = attn_mode
        self.cross_attn_2_type = attn_mode
    elif self.model_cls in ["hunyuan_video_1.5", "qwen_image", ...]:
        # 其他模型只有一种注意力
        self.attn_type = attn_mode
```

### 4.4 set_config()：配置整合

`set_config()` 是配置整合的真正入口，内部分两步完成：

```python
# set_config.py:135-138
def set_config(args):
    config = set_args2config(args)    # 第1步：默认配置 + Pipeline 属性
    config = auto_calc_config(config) # 第2步：加载模型配置 + 自动计算
    return config
```

**第1步：set_args2config() — 合并默认配置与 Pipeline 属性**

```python
# set_config.py:38-41
def set_args2config(args):
    config = get_default_config()  # 获取默认配置（LockableDict）
    # 把 Pipeline 的属性合并进来（排除 input_info 相关的 key）
    config.update({k: v for k, v in vars(args).items() if k not in ALL_INPUT_INFO_KEYS})
    return config
```

这里的 `args` 就是 Pipeline 实例本身（因为 `@dict_like` 让它支持 `vars()` 遍历属性）。`ALL_INPUT_INFO_KEYS` 过滤掉了 `seed`、`prompt` 等运行时输入参数，只保留配置类属性。

**第2步：auto_calc_config() — 加载模型配置并自动计算**

```python
# set_config.py:44-132
def auto_calc_config(config):
    # 1. 如果指定了 config_json，先加载
    if config.get("config_json", None) is not None:
        with open(config["config_json"], "r") as f:
            config.update(json.load(f))

    # 2. 根据 model_cls 从模型目录加载 config.json
    #    不同模型的目录结构不同，需要分别处理
    if config["model_cls"] in ["hunyuan_video_1.5", ...]:
        # HunyuanVideo：config 在 transformer 子目录下
        config["transformer_model_path"] = os.path.join(
            config["model_path"], "transformer", config["transformer_model_name"]
        )
        # 加载 transformer/xxx/config.json
        ...
    else:
        # 通用模型：依次尝试多个可能的 config.json 路径
        # model_path/config.json
        # model_path/low_noise_model/config.json
        # model_path/original/config.json
        # model_path/transformer/config.json
        ...

    # 3. 校正视频帧数（I2V 任务要求 num_frames-1 能被 vae_stride[0] 整除）
    if config["task"] in ["i2v", "s2v", "rs2v"]:
        if config["target_video_length"] % config["vae_stride"][0] != 1:
            config["target_video_length"] = config["target_video_length"] // config["vae_stride"][0] * config["vae_stride"][0] + 1

    # 4. 加载 VAE 配置，计算 vae_scale_factor
    if os.path.exists(os.path.join(config["model_path"], "vae", "config.json")):
        ...

    return config
```

**auto_calc_config 的关键职责**：
- 从模型目录加载 `config.json`（包含 `dim`、`num_heads`、`num_layers` 等网络结构参数）
- 处理不同模型的目录结构差异（HunyuanVideo 的 transformer 子目录、z_image 的 patch_size 特殊处理等）
- 自动校正不合法的参数（如视频帧数对齐）
- 加载量化权重的配置

**配置整合的完整流程**：

```
┌──────────────────────┐
│  ① 默认配置           │  get_default_config()
│  cpu_offload=False   │  feature_caching="NoCaching"
│  vae_stride=(4,8,8)  │  ...
└──────────┬───────────┘
           │  config.update(Pipeline属性)
           ▼
┌──────────────────────┐
│  ② Pipeline 属性覆盖  │  enable_xxx() 设置的参数
│  cpu_offload=True    │  dit_quantized=True
│  infer_steps=40      │  ...
└──────────┬───────────┘
           │  auto_calc_config()
           ▼
┌──────────────────────┐
│  ③ 模型 config.json   │  model_path/config.json
│  dim=4096            │  num_heads=64
│  num_layers=48       │  ...
└──────────┬───────────┘
           │  帧数校正、VAE配置等
           ▼
┌──────────────────────┐
│  ④ 最终 config        │  LockableDict
│  (传给 Runner)        │  包含所有配置
└──────────────────────┘
```

**注意优先级**：后加载的会覆盖先加载的。所以模型 config.json 中的参数会覆盖 Pipeline 设置的同名参数。这意味着模型自带的结构参数（如 `dim`、`num_heads`）始终以模型文件为准。

### 4.5 _init_runner()：创建 Runner

```python
# pipeline.py:445-449
def _init_runner(self, config):
    torch.set_grad_enabled(False)  # 推理模式，禁用梯度
    runner = RUNNER_REGISTER[config["model_cls"]](config)  # 通过注册器获取 Runner 类并实例化
    runner.init_modules()  # 初始化模块（加载模型）
    return runner
```

**这里就是注册机制发挥作用的地方**：
- `config["model_cls"]` = `"wan2.1"`
- `RUNNER_REGISTER["wan2.1"]` = `WanRunner` 类
- `WanRunner(config)` 创建实例

---

## 5. generate()：推理入口

```python
# pipeline.py:400-443
@torch.no_grad()
def generate(
    self,
    seed,                    # 随机种子
    prompt,                  # 提示词
    negative_prompt,         # 负面提示词
    save_result_path,        # 保存路径
    image_path=None,         # 输入图像（I2V 用）
    video_path=None,         # 输入视频（SR 用）
    last_frame_path=None,    # 最后一帧（FLF2V 用）
    audio_path=None,         # 音频路径（Audio2Video 用）
    return_result_tensor=False,  # 是否返回 tensor
    ...
):
    # 1. 保存输入参数到 Pipeline 属性
    self.seed = seed
    self.image_path = image_path
    self.prompt = prompt
    self.negative_prompt = negative_prompt
    self.save_result_path = save_result_path
    ...

    # 2. 创建 input_info 结构
    input_info = init_empty_input_info(self.task)
    seed_all(self.seed)  # 设置全局随机种子
    update_input_info_from_dict(input_info, self)  # 从 Pipeline 属性填充 input_info

    # 3. 调用 Runner 执行推理
    self.runner.run_pipeline(input_info)

    logger.info("Video generated successfully!")
    logger.info(f"Video Saved in {save_result_path}")
```

**input_info 的作用**：

`input_info` 是一个 dataclass，用于在 Pipeline 和 Runner 之间传递输入信息：

```python
# 不同任务有不同的 input_info 结构
@dataclass
class I2VInputInfo:
    seed: int
    prompt: str
    negative_prompt: str
    image_path: str
    save_result_path: str
    latent_shape: tuple
    ...

@dataclass
class T2VInputInfo:
    seed: int
    prompt: str
    negative_prompt: str
    save_result_path: str
    latent_shape: tuple
    ...
```

---

## 6. 完整调用流程

```
用户代码
    │
    ▼
pipe = LightX2VPipeline(task="i2v", model_cls="wan2.1", model_path="...")
    │
    │  设置模型特定的默认参数（vae_stride 等）
    │
    ▼
pipe.enable_offload(cpu_offload=True, ...)
pipe.enable_quantize(dit_quantized=True, ...)
    │
    │  收集优化配置到 Pipeline 属性
    │
    ▼
pipe.create_generator(infer_steps=40, height=480, width=832, ...)
    │
    ├── set_infer_config()  # 设置推理参数
    │
    ├── set_config(self)    # 整合所有配置
    │       │
    │       ├── get_default_config()  # 默认配置
    │       ├── Pipeline 属性         # enable_xxx 设置的
    │       └── model config.json     # 模型配置
    │
    ├── validate_config_paths()  # 验证路径
    │
    └── _init_runner(config)
            │
            ├── RUNNER_REGISTER["wan2.1"]  # 获取 WanRunner 类
            ├── WanRunner(config)          # 实例化
            └── runner.init_modules()      # 加载模型
    │
    ▼
pipe.generate(prompt="...", image_path="...", save_result_path="...")
    │
    ├── init_empty_input_info(task)  # 创建 input_info
    ├── seed_all(seed)               # 设置随机种子
    ├── update_input_info_from_dict()  # 填充 input_info
    │
    └── runner.run_pipeline(input_info)  # 执行推理
            │
            ├── run_input_encoder()  # 编码输入
            └── run_main()           # 去噪 + VAE 解码
```

---

## 7. 多任务支持的统一抽象

Pipeline 通过以下机制支持多种任务：

### 7.1 task 参数决定 input_info 结构

```python
# pipeline.py:128
self.input_info = init_empty_input_info(self.task)
```

不同 task 有不同的 input_info 字段：
- `t2v`：只需要 prompt
- `i2v`：需要 prompt + image_path
- `flf2v`：需要 prompt + image_path + last_frame_path
- `vace`：需要 prompt + src_video + src_mask

### 7.2 Runner 根据 task 绑定不同的编码器方法

```python
# default_runner.py:78-95
if self.config["task"] == "i2v":
    self.run_input_encoder = self._run_input_encoder_local_i2v
elif self.config["task"] == "t2v":
    self.run_input_encoder = self._run_input_encoder_local_t2v
elif self.config["task"] == "flf2v":
    self.run_input_encoder = self._run_input_encoder_local_flf2v
...
```

### 7.3 generate() 方法接受所有可能的输入

```python
def generate(
    self,
    seed,
    prompt,
    negative_prompt,
    save_result_path,
    image_path=None,       # I2V 用
    video_path=None,       # SR 用
    last_frame_path=None,  # FLF2V 用
    audio_path=None,       # Audio2Video 用
    src_video=None,        # VACE 用
    src_mask=None,         # VACE 用
    ...
):
```

用户只需要传入当前任务需要的参数，其他参数保持 None。

---

## 8. 关键源码文件索引

| 文件 | 行号 | 内容 |
|------|------|------|
| `pipeline.py` | 36-59 | @dict_like 装饰器 |
| `pipeline.py` | 62-128 | __init__ 初始化 |
| `pipeline.py` | 130-188 | create_generator |
| `pipeline.py` | 190-238 | set_infer_config |
| `pipeline.py` | 240-244 | set_infer_config_json |
| `pipeline.py` | 246-259 | enable_lightvae |
| `pipeline.py` | 261-297 | enable_quantize |
| `pipeline.py` | 299-336 | enable_offload |
| `pipeline.py` | 338-353 | enable_compile |
| `pipeline.py` | 355-367 | enable_lora / switch_lora |
| `pipeline.py` | 369-391 | enable_cache |
| `pipeline.py` | 393-398 | enable_parallel |
| `pipeline.py` | 400-443 | generate |
| `pipeline.py` | 445-449 | _init_runner |
| `set_config.py` | 15-35 | get_default_config |
| `set_config.py` | 38-41 | set_args2config |
| `set_config.py` | 44-132 | auto_calc_config（加载模型config.json、帧数校正等） |
| `set_config.py` | 135-138 | set_config 入口（set_args2config + auto_calc_config） |
| `set_config.py` | 141-168 | set_parallel_config |
| `set_config.py` | 44-100 | auto_calc_config |

---

## 9. 思考题

1. **为什么 enable_offload() 要根据 model_cls 设置不同的配置名？**
   - 提示：考虑不同模型使用的编码器不同

    不知道啊

    **参考答案**：因为不同模型使用的编码器完全不同：Wan 用 T5+CLIP，HunyuanVideo 用 Qwen2.5-VL+SigLIP，LTX2 用 Gemma，Z-Image 用 Qwen3。下游 Runner 加载编码器时，需要通过具体的配置名（如 `t5_cpu_offload`、`qwen25vl_cpu_offload`）来判断是否 offload 对应的编码器。如果统一用 `text_encoder_offload` 这个名字传到 Runner，Runner 还是得根据 model_cls 做一次判断。不如在 Pipeline 层就映射好，Runner 直接读自己关心的配置名即可。这是一种"**用户接口统一，内部配置具体化**"的设计。

2. **如果要添加一个新的任务类型（如 video2video），需要修改哪些地方？**
   - 提示：input_info、generate 参数、Runner 的 run_input_encoder

    一定需要：创建新的input_info类型，定义subRunner的新run_input_encoder方法

    可能需要：新的generate 参数

    **批改**：核心思路正确 ✅，但漏了几个必须改的地方：

    **必须修改的**：
    - `input_info.py`：新建 `V2VInputInfo` dataclass，并在 `init_empty_input_info()` 中添加 `elif task == "v2v"` 分支 ← 你答对了
    - `default_runner.py init_modules()`：添加 `elif self.config["task"] == "v2v": self.run_input_encoder = self._run_input_encoder_local_v2v`，绑定新的编码方法 ← 你答对了（"定义新的 run_input_encoder 方法"）
    - `pipeline.py generate()`：如果需要新的输入参数（如 `input_video_path`），需要添加参数并赋值到 `self` ← 你说"可能需要"，但对于 v2v 这种需要视频输入的任务，这是必须的
    - `input_info.py update_input_info_from_dict()`：确保新字段能从 Pipeline 属性正确填充到 input_info

    **可能需要的**：
    - 具体 Runner（如 WanRunner）中实现 `_run_input_encoder_local_v2v` 的具体编码逻辑
    - Pipeline `__init__` 中如果新任务有特殊的默认参数，需要添加

3. **为什么配置整合要分三层（默认 → Pipeline → 模型）？**
   - 提示：考虑灵活性和默认值

    不知道啊，感觉除了默认值要先初始化以外，剩下两个顺序颠倒一下也无所谓

    **批改**：默认值先初始化这个判断是对的 ✅，但"剩下两个顺序颠倒无所谓"不太对 ❌。

    实际的加载顺序是：默认配置 → Pipeline 属性 → 模型 config.json（看 `set_config` 的代码，`set_args2config` 先合并 Pipeline 属性，然后 `auto_calc_config` 再用模型 config.json 覆盖）。

    **为什么模型 config.json 要最后加载（优先级最高）？**

    模型 config.json 里存的是**网络结构参数**（`dim`、`num_heads`、`num_layers`、`vocab_size` 等），这些参数必须和模型权重严格匹配，不能被用户随意覆盖。如果反过来让 Pipeline 属性最后覆盖，用户可能不小心设了一个 `dim=2048`，但模型权重是按 `dim=4096` 训练的，加载时就会报 shape mismatch。

    所以三层的设计逻辑是：
    - **默认配置**：提供合理的兜底值，确保所有 key 都有值
    - **Pipeline 属性**：用户可控的运行时参数（推理步数、分辨率、是否量化等）
    - **模型 config.json**：模型自带的结构参数，不可被用户覆盖，保证权重和配置一致

4. **@dict_like 装饰器有什么好处？能否用其他方式实现？**
   - 提示：考虑 dataclass、namedtuple、继承 dict

    不知道啊

    **参考答案**：

    `@dict_like` 的好处是让 Pipeline 实例同时支持**属性访问**（`pipe.height`）和**字典访问**（`pipe["height"]`），这样 `set_args2config` 中的 `vars(args).items()` 就能把所有属性遍历出来合并到 config 字典中。

    替代方案对比：

    | 方案 | 优点 | 缺点 |
    |------|------|------|
    | `@dict_like`（当前） | 灵活，可以动态添加属性，不需要预定义字段 | 没有类型提示，IDE 补全差 |
    | `dataclass` | 有类型提示，IDE 友好 | 字段必须预定义，不能动态添加（enable_xxx 会动态加属性） |
    | `namedtuple` | 不可变，安全 | 不可变恰恰是缺点，Pipeline 需要不断修改属性 |
    | 继承 `dict` | 天然支持字典操作 | `self.key` 和 `self["key"]` 是两套存储，容易混乱 |

    当前方案最适合 Pipeline 的使用场景：**属性在 `__init__` 和各种 `enable_xxx()` 中动态添加**，无法提前穷举所有字段，所以 dataclass 和 namedtuple 都不太合适。

---

## 下一课预告

第3课将深入 Runner 层，分析：
- BaseRunner → DefaultRunner → 具体 Runner 的继承体系
- 组件初始化顺序与依赖关系
- WanRunner vs HunyuanVideoRunner 的差异
