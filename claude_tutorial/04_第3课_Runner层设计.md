# 第3课：Runner 层设计——生命周期、能力与模型族

> 源码基线：`LightX2V` `main@aad0aedbae93be1fba17b3090edb0b7c6f119600`。文中行号均对应此提交。

## 学习目标

完成本课后，你应当能够：

1. 用“生命周期基类、能力 Mixin、模型族”三条轴线定位任意 Runner，而不是只背继承树。
2. 从 `Pipeline/CLI → RUNNER_REGISTER → Runner → Scheduler/Model/Encoder/VAE` 追踪真实调用链。
3. 解释常驻、`unload_modules`、`lazy_load`、model/block/phase offload 之间的职责边界。
4. 读懂 Wan 的本地链与 encoder/transformer/decode 分离部署链。
5. 根据组件能力矩阵判断量化、卸载、并行、LoRA 是否真的由当前组件支持。
6. 完成新模型 Runner 适配，并同步检查 Pipeline、CLI、配置归一化和 InputInfo。

---

## 1. 不要只画一棵继承树：Runner 的三轴视图

Runner 层负责把模型族的组件装配为一次可运行请求：加载组件、编码输入、驱动采样循环、解码和后处理。当前代码更适合沿三条正交轴理解。

```text
                    Runner
                      │
      ┌───────────────┼────────────────┐
      │               │                │
生命周期轴          能力轴            模型族轴
BaseRunner          DisaggMixin       Wan / Qwen-Image
  └─ DefaultRunner  其他族内能力       Hunyuan / LTX2 / ...
      │               │                │
自动包装、warmup     可组合通信能力     组件、任务、调度器差异
GC、请求生命周期     不等于“父类层级”   决定覆盖哪些钩子
```

### 1.1 生命周期轴：`BaseRunner → DefaultRunner`

- `BaseRunner` 定义加载、编码、采样段、收尾等契约，并通过 `__init_subclass__` 自动包装入口。
- `DefaultRunner` 给出扩散生成的通用骨架：组件装配、按任务绑定输入编码函数、逐 step 调度、VAE 解码、保存输出。
- 继承 `DefaultRunner` 不代表所有方法都必须复用；特殊控制流可以覆盖 `run_pipeline()` 或 `run_main()`。

源码：

- `lightx2v/models/runners/base_runner.py:12-90`
- `lightx2v/models/runners/default_runner.py:95-156`
- `lightx2v/models/runners/default_runner.py:276-332`
- `lightx2v/models/runners/default_runner.py:433-516`

### 1.2 能力轴：Mixin 是“可选能力”，不是模型族

当前最明确的例子是 `DisaggMixin`：它提供 encoder 输出和 transformer latent 的发送/接收能力，不负责默认采样生命周期。

```python
class WanRunner(DisaggMixin, DefaultRunner):
    ...

class QwenImageRunner(DisaggMixin, DefaultRunner):
    ...
```

只有 `WanRunner` 和 `QwenImageRunner` 直接混入它；这不等于所有 `DefaultRunner` 都自动支持分离推理。Mixin 的初始化还要由具体 Runner 主动调用，例如 Wan 在 `init_modules()` 中根据 `disagg_mode` 调用 `init_disagg()`。

源码：

- `lightx2v/disagg/disagg_mixin.py:121-224`
- `lightx2v/models/runners/wan/wan_runner.py:75-86,396-400`
- `lightx2v/models/runners/qwen_image/qwen_image_runner.py:57-75`

### 1.3 模型族轴：组件和任务决定实现

模型族子类主要决定：

- Transformer、文本/图像编码器、VAE 和 Scheduler 用哪个实现；
- task 如何转成网络输入；
- CFG、分段、音频、世界状态等条件如何接入；
- 是否需要改变默认控制流。

`WorldMirrorRunner` 和 `FastWAMRunner` 直接继承 `BaseRunner`，是理解这条轴的关键反例：

- WorldMirror 是 3D 重建，没有扩散 Scheduler 的标准循环；
- FastWAM 是机器人动作策略，`run_pipeline()` 直接读取双视角图像和状态并输出 action chunk。

因此不能把“Runner”简单等同于“DefaultRunner 的视频扩散子类”。

源码：

- `lightx2v/models/runners/worldmirror/worldmirror_runner.py:131-148`
- `lightx2v/models/runners/wan/fastwam_runner.py:342-439`

---

## 2. 注册表事实：当前恰好 47 个实际 key

注册发生在模块 import 时；创建 Runner 时只是按 `config["model_cls"]` 查表并实例化。

```text
模块显式 import
  → @RUNNER_REGISTER("key") 执行
  → RUNNER_REGISTER[key] = RunnerClass
  → runner = RUNNER_REGISTER[config["model_cls"]](config)
```

创建入口：

- Pipeline：`lightx2v/pipeline.py:12-39,491-495`
- CLI：`lightx2v/infer.py:7-46,53-58`
- 注册器：`lightx2v/utils/registry_factory.py:6-65,76`

### 2.1 按家族概括 47 个 key

| 家族 | 数量 | 当前实际注册 key |
|---|---:|---|
| Wan 及其派生机器人/世界模型 | 23 | `wan2.1`、`wan2.2`、`wan2.2_moe`、`wan2.1_vace`、`wan2.2_moe_vace`、`wan2.1_distill`、`wan2.1_mean_flow_distill`、`wan2.2_moe_distill`、`wan2.1_sf`、`wan2.1_sf_mtxg2`、`wan2.2_animate`、`wan2.2_s2v`、`seko_talk`、`wan2.2_audio`、`seko_talk_ar`、`wan_dancer`、`dreamzero`、`infinitetalk`、`wan2.2_matrix_game3`、`lingbot_world`、`lingbot_world_fast`、`lingbot_va`、`motus` |
| 其他视频/世界生成 | 10 | `hunyuan_video_1.5`、`hunyuan_video_1.5_distill`、`worldplay_distill`、`worldplay_ar`、`worldplay_bi`、`ltx2`、`ltx2_ar`、`cosmos3`、`lingbot_video`、`neopp` |
| 图像与统一多模态生成 | 10 | `qwen_image`、`z_image`、`ernie_image`、`ernie_image_turbo`、`flux2_klein`、`flux2_dev`、`longcat_image`、`hunyuan_image3`、`hidream_o1_image`、`bagel` |
| 独立任务 Runner | 4 | `seedvr2`、`hunyuan3d`、`worldmirror`、`fastwam` |
| **合计** | **47** | 以 `@RUNNER_REGISTER(...)` 的实际出现为准 |

完整注册位置分散在 36 个 Runner 文件；可从 `lightx2v/models/runners/wan/wan_runner.py:75,968,1067,1195` 等装饰器位置开始追踪。

### 2.2 三个容易写错的“独立 key”

- Animate 是 `wan2.2_animate`，不是旧的 `wan2.1_animate`：`lightx2v/models/runners/wan/wan_animate_runner.py:29`。
- SeedVR 是 `seedvr2`，不是 `seedvr`：`lightx2v/models/runners/seedvr/seedvr_runner.py:75`。
- WorldPlay 是三个独立 key：`worldplay_distill`、`worldplay_ar`、`worldplay_bi`，不是一个 `worldplay_*` 通配注册：
  - `lightx2v/models/runners/worldplay/worldplay_distill_runner.py:18`
  - `lightx2v/models/runners/worldplay/worldplay_ar_runner.py:18`
  - `lightx2v/models/runners/worldplay/worldplay_bi_runner.py:93`

### 2.3 `wan2.2_moe_audio` 是当前不一致项

`wan2.2_moe_audio` 出现在 Pipeline 的 Wan 分支和 CLI choices 中：

- `lightx2v/pipeline.py:94-110,339-355`
- `lightx2v/infer.py:88-128`

但 47 个装饰器中没有 `@RUNNER_REGISTER("wan2.2_moe_audio")`。实际音频注册是：

- `seko_talk`：`lightx2v/models/runners/wan/wan_audio_runner.py:285`
- `wan2.2_audio`：`lightx2v/models/runners/wan/wan_audio_runner.py:947`
- `seko_talk_ar`：`lightx2v/models/runners/wan/wan_audio_runner.py:994`

因此 CLI 目前会接受 `wan2.2_moe_audio`，随后在 `RUNNER_REGISTER[...]` 查找阶段失败。适配或排障时必须比较“注册表、import、CLI choices、Pipeline 分支”，不能只看其中一处。

### 2.4 蒸馏 CFG 是配置，不是 Runner 强制规则

`WanDistillRunner` 只替换模型和 Scheduler，并没有在类中强制关闭 CFG：`lightx2v/models/runners/wan/wan_distill_runner.py:13-33`。仓库多数蒸馏配置把 `enable_cfg` 设为 `false`，例如：

- `configs/distill/wan21/wan_t2v_distill_model_4step_cfg.json:12`
- `configs/distill/wan22/wan_moe_i2v_distill_model.json:15`

所以正确表述是“当前蒸馏配置通常关闭 CFG”，而不是“蒸馏 Runner 天生无 CFG”。网络层仍按 `config["enable_cfg"]` 决定是否执行条件/无条件前向。

---

## 3. 当前生命周期：自动包装、warmup、GC 与请求后卸载

### 3.1 类定义时自动包装

`BaseRunner.__init_subclass__()` 在子类定义完成时做两类包装：

```text
子类定义 init_modules
  → 维护嵌套 depth
  → 仅最外层 init_modules 返回后调用一次 warmup

子类定义 run_pipeline
  → 原 run_pipeline 返回
  → 调用一次 _maybe_freeze_gc
```

`depth` 解决了 `WanRunner.init_modules() → DefaultRunner.init_modules()` 的嵌套调用：否则每层都可能 warmup。`_warmup_done` 和 `_gc_frozen` 都是一实例一次性门闩。

源码：`lightx2v/models/runners/base_runner.py:18-57`。

### 3.2 warmup 的真实边界

- `BaseRunner.warmup()`：若用户显式要求 warmup 而子类没有实现，直接报错。
- `DefaultRunner.warmup()`：拒绝 disagg、`unload_modules` 和 feature caching，再调用模型族 `run_warmup()`。
- Wan 只允许声明支持的子类和 `t2v/i2v/flf2v`；对两组分辨率运行代表性 step，并执行 VAE 解码。
- `lazy_load` warmup 会临时建模，结束后清掉 offload manager、模型和编码组件。

源码：

- `lightx2v/models/runners/base_runner.py:59-63`
- `lightx2v/models/runners/default_runner.py:100-113`
- `lightx2v/models/runners/wan/wan_runner.py:87-214`

### 3.3 `gc.freeze()` 何时发生

普通常驻模式在第一次成功返回 `run_pipeline()` 后：

1. `gc.collect()` 清理循环垃圾；
2. 统计当前活对象；
3. `gc.freeze()` 把稳定对象图放进永久代，减少后续 GC 扫描。

若开启 `lazy_load` 或 `unload_modules`，对象图每个请求都会重建，代码会跳过 freeze，但仍把 `_gc_frozen` 置为真，避免重复判断和日志。

注意：包装器在原 `run_pipeline()` 正常返回后才执行；若请求抛异常，不会到达 freeze。

源码：`lightx2v/models/runners/base_runner.py:65-83`。

### 3.4 常驻、`unload_modules`、`lazy_load` 不完全相同

| 模式 | `init_modules()` | 请求中 | 请求结束 |
|---|---|---|---|
| 常驻 | `load_model()` 一次装入全部组件 | 直接复用 | 清 Scheduler/输入，模型保留 |
| `unload_modules=true` | 不预加载组件 | 各阶段临时从普通 checkpoint 构建 | 删除临时编码器、VAE、Transformer |
| `lazy_load=true` | 不预加载，且要求 `cpu_offload=true` | 除阶段临时构建外，Transformer block 可由拆分 checkpoint 按需从磁盘预取 | 删除临时模型/offload manager |

`lazy_load` 是更具体的磁盘按需能力；`unload_modules` 是请求级不常驻策略。代码经常用二者的 OR 处理组件生命周期，但只有 `lazy_load` 会让 `BaseTransformerModel` 寻找 `non_block.safetensors` 并建立磁盘预取缓冲。

源码：

- `lightx2v/models/runners/default_runner.py:115-128,276-316,433-505`
- `lightx2v/models/networks/base_model.py:427-475`
- `lightx2v/common/offload/manager.py:122-159`

---

## 4. 真实调用链：从 API 到一次 DiT step

### 4.1 初始化链

```text
LightX2VPipeline.create_generator()
  ├─ set_config(self) + validate_config_paths()
  ├─ 可选 set_parallel_config()
  └─ _init_runner(config)
       ├─ RUNNER_REGISTER[model_cls](config)
       │    └─ Runner.__init__(): set_init_device + init_scheduler
       └─ runner.init_modules()
            ├─ [常驻] load_model()
            │    ├─ load_transformer()
            │    ├─ load_text_encoder()
            │    ├─ load_image_encoder()
            │    └─ load_vae()
            ├─ model.set_scheduler(scheduler)
            ├─ 按 task 绑定 run_input_encoder
            ├─ config.lock()
            └─ [自动包装] warmup()
```

源码：`lightx2v/pipeline.py:175-205,491-495`，`lightx2v/models/runners/default_runner.py:97-156`。

### 4.2 请求链

```text
Pipeline.generate()
  ├─ init_empty_input_info(task, support_tasks)
  ├─ update_input_info_from_dict(input_info, pipeline)
  └─ runner.run_pipeline(input_info)
       ├─ 可选 prompt enhancer
       ├─ run_input_encoder()
       └─ run_main()
            ├─ init_run(): 必要时加载 Transformer，scheduler.prepare()
            ├─ 对每个 segment
            │    ├─ init_run_segment()
            │    ├─ run_segment()
            │    │    └─ 对每个 step
            │    │         ├─ scheduler.step_pre(step_index)
            │    │         ├─ model.infer(inputs)
            │    │         └─ scheduler.step_post()
            │    ├─ VAE decode
            │    └─ end_run_segment()
            ├─ process_images_after_vae_decoder()
            └─ end_run()
```

源码：

- `lightx2v/pipeline.py:425-489`
- `lightx2v/models/runners/default_runner.py:276-316,420-516,579-596`

---

## 5. Wan：本地链与 encoder/transformer/decode 分离链

`WanRunner(DisaggMixin, DefaultRunner)` 覆盖 `load_model()` 和 `run_pipeline()`，使同一个 Runner 能按角色裁剪组件。

### 5.1 本地模式

```text
输入
  → T5 / 可选 CLIP / 可选 VAE encode
  → Scheduler.prepare
  → WanModel.infer × infer_steps
  → VAE decode
  → wan_vae_to_comfy
  → tensor 或文件
```

本地模式 `_run_pipeline_local()` 仍复用默认的 `run_main()`，并没有复制一套采样循环。

源码：`lightx2v/models/runners/wan/wan_runner.py:455-458,481-507`。

### 5.2 分离模式

```text
encoder 角色
  load: T5 + CLIP + VAE encoder/decoder
  run : 编码输入 → send_encoder_outputs(inputs, latent_shape)
                         │
                         ▼
transformer 角色
  load: DiT；有独立 decoder 时不加载 VAE
  run : receive_encoder_outputs → 去噪 → send_transformer_outputs(latents)
                         │
                         ▼
decode 角色
  load: 仅 VAE decoder
  run : receive_transformer_outputs → VAE decode → 后处理/保存
```

若 transformer 没配置独立 decoder，它会在本地完成 VAE decode。`BaseRunner.init_scheduler()` 还会为 decode 角色设置 `NullScheduler`，避免装入真实扩散调度器。

源码：

- 角色化加载：`lightx2v/models/runners/wan/wan_runner.py:402-442`
- 角色化运行：`lightx2v/models/runners/wan/wan_runner.py:444-507`
- P1 发送/接收：`lightx2v/disagg/disagg_mixin.py:693-941`
- P2 发送/接收：`lightx2v/disagg/disagg_mixin.py:1050-1167`
- Decode NullScheduler：`lightx2v/models/runners/base_runner.py:192-197`

分离能力来自 Mixin，但“各角色装哪些组件、在哪条链调用发送/接收”仍由 WanRunner 决定。

---

## 6. Wan 输入编码：顺序调度与 VAE 内部并行要分清

### 6.1 I2V 当前是顺序调用，不是 CLJosh/CLIP 与 VAE 并发

`DefaultRunner._run_input_encoder_local_i2v()` 的顺序是：

```text
read_image_input
  → run_image_encoder（可选 CLIP）
  → run_vae_encoder
  → run_text_encoder
  → get_encoder_output_i2v
```

最终契约：

```python
{
    "text_encoder_output": text_encoder_output,
    "image_encoder_output": {
        "clip_encoder_out": clip_encoder_out,
        "vae_encoder_out": vae_encoder_out,
    },
}
```

T2V 则直接计算 latent shape、运行文本编码器，并令 `image_encoder_output=None`。

源码：

- `lightx2v/models/runners/default_runner.py:386-410`
- `lightx2v/models/runners/wan/wan_runner.py:844-870`

### 6.2 VAE 可以在分布式 rank 间并行

这与上面的组件调用顺序是另一层概念。Wan VAE 的 `parallel` 来自：

- tensor parallel 开启时强制关闭 VAE parallel；
- `parallel` 为布尔值时直接使用；
- `parallel` 为字典时读 `parallel.vae_parallel`，默认真。

Wan 2.1 编码路径可根据 latent 高宽选择二维 rank 网格，对输入加边缘 padding 后由各 rank 调用 `encode_local_2d()`；不满足整除约束会显式报错。也就是说，**一次 VAE encode 内部可并行**，但 CLIP、VAE、T5 在默认 Runner 中仍按顺序发起。

源码：

- VAE parallel 配置：`lightx2v/models/runners/wan/wan_runner.py:316-323`
- 网格与局部输入：`lightx2v/models/runners/wan/wan_runner.py:604-757`
- 编码分支：`lightx2v/models/runners/wan/wan_runner.py:793-842`

### 6.3 VAE 不是统一量化组件

Pipeline 的 `enable_quantize()` 当前覆盖 DiT、文本编码器和图像编码器，没有通用 `vae_quantized` 接口：`lightx2v/pipeline.py:290-326`。Wan VAE 当前可配置的主要能力是：

- `vae_cpu_offload`；
- `vae_dtype`（Wan 2.1 路径）；
- VAE parallel、tiling；
- LightVAE / TAE 或 Wan 2.2 的模型族特定 VAE 类型。

不要由“组件加载模式相似”推导出“所有组件都统一支持量化”。

### 6.4 `clip_cpu_offload` 与 Pipeline 当前不一致

WanRunner 实际读取：

```python
clip_offload = config.get("clip_cpu_offload", config.get("cpu_offload", False))
```

但 `LightX2VPipeline.enable_offload()` 写入的是 `self.clip_encoder_offload`，不是 `clip_cpu_offload`。

- Runner：`lightx2v/models/runners/wan/wan_runner.py:230-259`
- Pipeline：`lightx2v/pipeline.py:328-357`

因此通过当前 Pipeline API 传 `image_encoder_offload=True`，Wan CLIP 不一定按预期读取到；它可能回退到全局 `cpu_offload`。disagg 工具写的却是 `clip_cpu_offload`：`lightx2v/disagg/utils.py:77`。这是实际命名不一致，教程不能把它描述成已经完成的映射。

---

## 7. 组件能力矩阵

“支持某能力”必须落到具体组件和实现，不应只看 Runner 顶层配置。

| Runner/组件 | Transformer | 文本/图像编码 | VAE/输出 | 关键能力边界 |
|---|---|---|---|---|
| Wan | `WanModel`，含单模型和双专家代理 | T5；I2V 等可用 CLIP | Wan2.1/Wan2.2 VAE，可选 TAE/LightVAE | DiT 多量化、TP/SP/CFG、block/phase/model offload；VAE 无统一量化开关 |
| Qwen-Image | `QwenImageTransformerModel` | Qwen2.5-VL 文本侧 | `AutoencoderKLQwenImageVAE` | 混入 DisaggMixin；图像任务有自身输入/尺寸契约 |
| HunyuanVideo 1.5 | `HunyuanVideo15Model` | Qwen2.5-VL + ByT5；SigLIP | Hunyuan VAE | 仅 double-stream blocks；蒸馏是派生 Runner |
| LTX2 | `LTX2Model/LTX2ARModel` | Gemma 等族内组件 | 视频与音频 VAE/声码器链 | t2av/i2av/v2av/AR，模型输出和 Scheduler 状态不止单个视频张量 |
| SeedVR2 | `SeedVRNaDiTModel` | SR 条件预处理 | SR 解码/后处理 | 实际 key 为 `seedvr2`，调度器使用 velocity prediction |
| WorldMirror | 自有 `WorldMirrorWeightModel` | 多视图重建输入 | 深度/法线/点/GS 等输出 | 直接 BaseRunner；无默认扩散生命周期 |
| FastWAM | `FastWAMPolicy` | 双视角图像 + 机器人状态 + 文本任务 | action chunk | 直接 BaseRunner；无 VAE/扩散 Scheduler |

代表源码：

- `lightx2v/models/runners/wan/wan_runner.py:216-394`
- `lightx2v/models/runners/qwen_image/qwen_image_runner.py:57-75`
- `lightx2v/models/runners/hunyuan_video/hunyuan_video_15_runner.py:29-116`
- `lightx2v/models/runners/ltx2/ltx2_runner.py:89-160`
- `lightx2v/models/runners/seedvr/seedvr_runner.py:75-103`

---

## 8. MultiModelStruct：双模型的 lazy/unload 与单模型 model offload

### 8.1 对外保持单模型接口

Wan 2.2 MoE 用 `MultiModelStruct` 包装 `[high_noise_model, low_noise_model]`。它对 DefaultRunner 暴露 `set_scheduler()` 和 `infer()`，内部按 timestep 边界选择专家，并同步切换 `sample_guide_scale`。

```text
DefaultRunner
  └─ self.model.infer(inputs)
       └─ MultiModelStruct
            ├─ timestep >= boundary → high expert
            └─ timestep <  boundary → low expert
```

源码：`lightx2v/models/runners/wan/wan_runner.py:873-965`。

### 8.2 三种容易混淆的行为

1. **双模型 + 常驻，非 model offload**：两个专家对象都已创建；底层可各自使用 block/phase offload。
2. **双模型 + `offload_granularity="model"`**：两个专家对象都存在，切换边界时把整个旧专家搬到 CPU、把新专家搬到设备；不是销毁重建。
3. **双模型 + `lazy_load` 或 `unload_modules`**：`load_transformer()` 先返回 `[None, None]` 的代理；第一次命中某专家时才创建该 `WanModel`。请求结束后 DefaultRunner 删除整个代理及其 offload manager。

源码：

- 运行时建专家：`lightx2v/models/runners/wan/wan_runner.py:891-934`
- model 级专家切换：`lightx2v/models/runners/wan/wan_runner.py:937-965`
- MoE 加载：`lightx2v/models/runners/wan/wan_runner.py:968-1035`
- 请求结束卸载：`lightx2v/models/runners/default_runner.py:292-320`

单个 `WanModel` 的 model offload 不经过 `MultiModelStruct`：第一个 step 整模 `to_cuda()`，最后一个 step 整模 `to_cpu()`；block/phase 模式则每次 infer 只把 pre/non-block 权重整体搬运，block 主体由异步 offload manager 调度。

源码：`lightx2v/models/networks/wan/model.py:269-317`。

---

## 9. 输出转换与 LoRA：两个常见契约陷阱

### 9.1 `wan_vae_to_comfy()` 返回浮点张量

它原地执行 `[-1,1] → [0,1]` 的浮点缩放、clamp 和维度重排：

- 视频：`[B,C,T,H,W] → [B*T,H,W,C]`；
- 图像：`[B,C,H,W] → [B,H,W,C]`。

它没有转成 `uint8`。文件保存函数可以在后续编码时量化，但 `return_result_tensor=True` 返回的是浮点 tensor。

源码：`lightx2v/utils/utils.py:179-205`，调用点 `lightx2v/models/runners/default_runner.py:523-545`。

### 9.2 运行时切换 LoRA 的前置条件

Pipeline 层先检查 `lora_dynamic_apply`：未启用就不会调用 Runner。Runner 层还要求：

- 模型当前已加载；
- 模型有 `_update_lora`；
- 删除时还要有 `_remove_lora`。

对 `lazy_load/unload_modules`，请求外模型可能为 `None` 或已删除，不能假设随时可切。Wan2.2 MoE 还要求目标专家对象已经实际创建；若仍是 lazy placeholder，当前 `switch_lora()` 只会跳过该专家。

初始化 LoRA 还有两条路径：

- 动态：把路径/强度交给模型，在 WeightModule 中注册可更新分支；
- 合并：要求非量化且非 `lazy_load`，由 `LoraAdapter` 合入权重。

源码：

- Pipeline 前置检查：`lightx2v/pipeline.py:385-398`
- Runner 运行时切换：`lightx2v/models/runners/default_runner.py:598-693`
- Wan 构建约束：`lightx2v/models/runners/wan/wan_runner.py:47-72`
- MoE 切换：`lightx2v/models/runners/wan/wan_runner.py:1037-1064`

---

## 10. 新 Runner 适配清单

只写一个 `@RUNNER_REGISTER` 类通常不够。按以下顺序核对。

### 10.1 先选择生命周期基类和能力

- 标准扩散/流匹配采样：优先复用 `DefaultRunner`。
- 只需额外通信能力：在确认 MRO 后混入已有 Mixin。
- 控制流完全不同：像 WorldMirror/FastWAM 一样直接实现 BaseRunner 契约。
- 不要为了一个模型复制 `run_segment()`；只有循环结构本身改变时才覆盖 `run_main()`/`run_pipeline()`。

### 10.2 定义组件和输入契约

至少核对：

- `load_transformer()`、`load_text_encoder()`、`load_image_encoder()`、`load_vae()`；
- `init_scheduler()`；
- task 到 `run_input_encoder` 的绑定；
- 编码输出的 key、shape、dtype 是否与 Model/Infer 一致；
- 是否真的支持 CFG、SP/TP、量化、offload、lazy、warmup、LoRA。

### 10.3 同时修改或确认五个集成面

1. **注册模块本身**：`@RUNNER_REGISTER("new_key")`。
2. **显式 import**：
   - API 入口 `lightx2v/pipeline.py:12-39`；
   - CLI 入口 `lightx2v/infer.py:7-46`。
3. **CLI choices**：`lightx2v/infer.py:88-135`，否则命令行到不了注册表；也要避免 `wan2.2_moe_audio` 这种 choices 有而注册表无的情况。
4. **Pipeline 分支**：`lightx2v/pipeline.py:77-145,252-357`，核对 stride、latent channel、attention key、量化和 offload key；若只支持 JSON，也应明确这一边界。
5. **配置与输入**：
   - `lightx2v/utils/set_config.py:13-239` 中的默认值、模型配置读取和归一化；
   - `lightx2v/utils/input_info.py:14-509` 中 task dataclass、`task_dict`、多 task 合并字段；
   - CLI/Pipeline 是否把新增字段传进 InputInfo。

### 10.4 最小验证矩阵

- 导入后断言 `new_key in RUNNER_REGISTER`；
- CLI choices 与注册表比较；
- 初始化常驻模式并跑一个最小 step；
- 若宣称支持，再分别测试 CFG、offload、lazy、量化、LoRA、并行；
- 检查 tensor 返回和文件保存两条输出路径；
- 失败组合应显式报 `NotImplementedError`，不要静默回退。

---

## 11. 常见误区

1. **“所有 Runner 都是 DefaultRunner。”** 错。WorldMirror 和 FastWAM 直接继承 BaseRunner。
2. **“Mixin 就是继承树中间层。”** 错。它表达可组合能力；当前 WanRunner、QwenImageRunner 显式混入 DisaggMixin。
3. **“注册了类，Pipeline 和 CLI 会自动发现。”** 错。装饰器依赖模块被显式 import，CLI choices 还是另一道门。
4. **“WorldPlay 有一个通配 key。”** 错。当前是三个独立注册键。
5. **“蒸馏 Runner 强制关闭 CFG。”** 错。多数配置关闭，但 Runner 逻辑仍读取配置。
6. **“所有组件都有统一量化接口。”** 错。当前 Pipeline 通用量化入口不含 VAE。
7. **“Wan 输入编码器并发执行。”** 错。默认顺序是 CLIP、VAE、T5；VAE 自身可做 rank 间并行。
8. **“Pipeline 的 image_encoder_offload 已正确映射给 Wan CLIP。”** 当前不成立：写 `clip_encoder_offload`，读 `clip_cpu_offload`。
9. **“lazy_load 与 unload_modules 完全同义。”** 错。二者都可请求级重建，但 lazy 还涉及拆分 checkpoint 和磁盘预取。
10. **“`wan_vae_to_comfy` 输出 uint8。”** 错。输出是 `[0,1]` 浮点张量。
11. **“开启 LoRA 配置后可在任意时刻 switch。”** 错。动态模式、已加载模型和对应更新接口都是前置条件。

---

## 12. 源码索引

| 主题 | 源码路径:当前行号 |
|---|---|
| BaseRunner 自动包装、warmup、GC | `lightx2v/models/runners/base_runner.py:18-90` |
| BaseRunner 生命周期契约 | `lightx2v/models/runners/base_runner.py:136-228` |
| DefaultRunner 初始化与任务绑定 | `lightx2v/models/runners/default_runner.py:95-156` |
| 默认组件装配 | `lightx2v/models/runners/default_runner.py:145-156` |
| 默认 step 循环与 end_run | `lightx2v/models/runners/default_runner.py:276-332` |
| I2V/T2V 输入编码 | `lightx2v/models/runners/default_runner.py:386-410` |
| 默认 run_main/VAE decode | `lightx2v/models/runners/default_runner.py:420-505` |
| 后处理与 run_pipeline | `lightx2v/models/runners/default_runner.py:517-596` |
| 运行时 LoRA | `lightx2v/models/runners/default_runner.py:598-693` |
| Wan LoRA 构建约束 | `lightx2v/models/runners/wan/wan_runner.py:47-72` |
| Wan warmup | `lightx2v/models/runners/wan/wan_runner.py:87-214` |
| Wan T5/CLIP/VAE 加载 | `lightx2v/models/runners/wan/wan_runner.py:216-394` |
| Wan 分离角色加载/调用 | `lightx2v/models/runners/wan/wan_runner.py:402-507` |
| Wan 编码执行 | `lightx2v/models/runners/wan/wan_runner.py:509-586,758-870` |
| Wan VAE 2D parallel | `lightx2v/models/runners/wan/wan_runner.py:604-842` |
| MultiModelStruct / MoE | `lightx2v/models/runners/wan/wan_runner.py:873-1064` |
| Wan2.2 Dense VAE | `lightx2v/models/runners/wan/wan_runner.py:1067-1192` |
| DisaggMixin 初始化 | `lightx2v/disagg/disagg_mixin.py:121-224` |
| Encoder/Transformer 传输 | `lightx2v/disagg/disagg_mixin.py:693-941,1050-1167` |
| Pipeline 模型分支与能力开关 | `lightx2v/pipeline.py:77-145,252-398` |
| CLI imports/choices | `lightx2v/infer.py:7-46,88-147` |
| InputInfo 与 task_dict | `lightx2v/utils/input_info.py:14-509` |
| 配置归一化 | `lightx2v/utils/set_config.py:13-239` |
| 输出浮点转换 | `lightx2v/utils/utils.py:179-205` |

---

## 13. 思考题

1. 为什么 `BaseRunner.__init_subclass__()` 需要 `_init_modules_depth`，只用 `_warmup_done` 是否足够？
2. 如果一个新 Runner 只支持 encoder/transformer 两段分离、不支持独立 decode，应该复用 DisaggMixin 的哪些接口，在哪个 Runner 方法中裁剪组件？
3. `wan2.2_moe_audio` 的问题为什么不能只在 CLI choices 中删除或只补一个注册装饰器就草率结束？还应检查哪些配置和组件语义？
4. 在 Wan I2V 中，如何区分“编码组件之间并发”和“VAE 单组件内部多 rank 并行”？请指出两者各自的调度位置。
5. `MultiModelStruct` 的 lazy placeholder 与 model offload 都能降低设备常驻，为什么它们的首次请求延迟和 CPU 内存行为不同？
6. 若希望运行时切换 MoE 两个专家的 LoRA，如何处理尚未由 lazy 路径实例化的专家，才能避免当前静默跳过？
7. 新增 `model_cls` 后，怎样写一个自动检查，保证注册表、CLI choices 和入口 import 不再出现单边漂移？
