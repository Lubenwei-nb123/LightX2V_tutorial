# 第 1 课：`lightx2v_train.trainers` 的设计和作用

> 代码版本：`main` 分支，`16d72024 add build image script (#1203)`。
> 说明：当前最新代码里没有名为 `lightx2v.trainer` 的 Python 包；训练相关实现位于 `lightx2v_train/lightx2v_train/trainers`。下文按这个真实模块分析。

## 1. 这个模块在整体链路中的位置

训练入口是 `lightx2v_train/train.py`，主流程非常短：

```text
load_config
  -> init_distributed / setup_logger
  -> build_model(config) + model.load_components()
  -> build_data(config, train) / build_data(config, val)
  -> build_trainer(config)
  -> trainer.set_model(model)
  -> trainer.set_data(train_loader, val_loader)
  -> trainer.train()
```

关键代码位置：

- `lightx2v_train/train.py:18` 读取 YAML 配置。
- `lightx2v_train/train.py:25` 通过 `build_model(config)` 构建模型适配器。
- `lightx2v_train/train.py:29` / `lightx2v_train/train.py:30` 构建训练和验证 dataloader。
- `lightx2v_train/train.py:32` 通过 `build_trainer(config)` 构建训练器。
- `lightx2v_train/train.py:33` / `lightx2v_train/train.py:34` 把模型和数据注入训练器。
- `lightx2v_train/train.py:36` 把控制权交给 `trainer.train()`。

所以，`trainer` 不是模型本身，也不是数据集本身，而是把 **模型、数据、优化器、分布式、checkpoint、训练算法、训练期推理验证** 串起来的调度层。

## 2. 注册机制：由配置选择训练算法

LightX2V 训练代码使用一个简单 Registry：

- `TRAINER_REGISTER` 定义在 `lightx2v_train/lightx2v_train/utils/registry.py:47`。
- `build_trainer(config)` 读取 `config["training"]["method"]`，再从注册表里取对应类，见 `lightx2v_train/lightx2v_train/utils/registry.py:70`。
- 各训练器通过装饰器注册，例如 `@TRAINER_REGISTER("flow")` 位于 `lightx2v_train/lightx2v_train/trainers/flow.py:12`。
- `lightx2v_train/lightx2v_train/trainers/__init__.py:3` 到 `lightx2v_train/lightx2v_train/trainers/__init__.py:6` 负责 import 各训练器模块，触发注册副作用。

当前训练器注册名：

| `training.method` | 类 | 文件 | 主要用途 |
|---|---|---|---|
| `flow` | `FlowMatchingTrainer` | `trainers/flow.py` | 通用 flow matching 训练，支持 LoRA / full。 |
| `teacher_forcing` | `TFTrainer` | `trainers/tf.py` | Wan causal teacher-forcing，全参训练。 |
| `dmd` | `DmdTrainer` | `trainers/dmd.py` | 图像模型 DMD 蒸馏，student/fake/teacher 三模型。 |
| `video_dmd` | `VideoDmdTrainer` | `trainers/dmd.py` | Wan 双向视频 DMD。 |
| `video_ar_dmd` | `VideoArDmdTrainer` | `trainers/dmd.py` | Wan 自回归 causal 视频 DMD。 |
| `dopsd` | `DopsdTrainer` | `trainers/dopsd.py` | D-OPSD 风格双 LoRA student/teacher EMA 训练。 |

这是一种典型的 Strategy Pattern：配置决定训练策略，入口不需要写 if/else 分支。

## 3. `BaseTrainer` 的职责边界

`BaseTrainer` 位于 `lightx2v_train/lightx2v_train/trainers/base.py:18`。它做的是训练器公共基础设施，不实现具体 loss。

### 3.1 配置解析

初始化时它读取：

- `model` / `training` / `inference` 三块配置，见 `base.py:20` 到 `base.py:23`。
- `scheduler` 配置，构造 `RectifiedFlowMatchingScheduler`，见 `base.py:25`。
- `running_dtype`，见 `base.py:26`。
- `training.train_type`，只允许 `lora` 或 `full`，见 `base.py:61` 到 `base.py:67`。
- LoRA 参数、optimizer 参数、lr scheduler、训练步数、checkpoint 间隔、日志间隔、auto-resume 等，见 `base.py:30` 到 `base.py:59`。

这里体现了一个重要设计：训练行为基本由 YAML 驱动，而不是在入口脚本里硬编码。

### 3.2 可训练参数选择

`BaseTrainer._setup_trainable_model()` 根据 `train_type` 决定训练方式：

- `lora`：调用 `model.add_lora()` 和 `model.set_lora_trainable()`。
- `full`：调用 `model.set_full_trainable()`。

对应代码在 `lightx2v_train/lightx2v_train/trainers/base.py:82` 到 `base.py:87`。

模型侧的默认实现位于 `lightx2v_train/lightx2v_train/model_zoo/base.py`：

- `add_lora()`：`base.py:33`。
- `set_lora_trainable()`：`base.py:100`。
- `set_full_trainable()`：`base.py:107`。
- `trainable_parameters()`：`base.py:112`。

也就是说，训练器不直接理解某个具体模型内部结构，它只要求模型适配器暴露统一方法。

### 3.3 分布式与显存策略

`BaseTrainer.setup()` 中会调用 `apply_parallel(self.model, self.config)`，见 `lightx2v_train/lightx2v_train/trainers/base.py:123`。

并行策略在 `lightx2v_train/lightx2v_train/runtime/parallel.py`：

- `apply_parallel()` 优先检查是否分布式，见 `parallel.py:8` 到 `parallel.py:21`。
- 如果配置启用 DP/DDP，则走 `apply_ddp()`。
- 如果配置启用 FSDP2，则走 `apply_fsdp2()`。
- 梯度累积时用 `set_parallel_gradient_sync()` 控制是否同步梯度，见 `parallel.py:24` 到 `parallel.py:26`。

顺序很关键：先根据 `train_type` 设置 LoRA 或 full 的可训练参数，再包 DDP/FSDP2。这样分布式包装面对的是已经明确 trainable/frozen 状态的 denoiser。

### 3.4 optimizer / lr scheduler / inference hook

`setup()` 还负责：

- 开启 gradient checkpointing，见 `base.py:125`。
- 如果配置了 `inference.infer_every_iters`，构造 inferencer，见 `base.py:128` 到 `base.py:130`。
- 记录模型结构，见 `base.py:132`。
- 收集 `trainable_params`，构造 optimizer 和 lr scheduler，见 `base.py:134` 到 `base.py:136`。
- 如果 auto-resume 找到 checkpoint，则恢复训练状态，见 `base.py:138` 到 `base.py:139`。

训练期推理在 `run_inference()` 中执行，见 `base.py:227` 到 `base.py:239`。推理结束后会调用 `_restore_trainable_model()`，避免推理把模型留在 eval 或错误 adapter 状态。

### 3.5 checkpoint / resume

`BaseTrainer` 支持两类保存：

- 普通单进程或 DDP：保存独立模型权重和 `training_state.pt`，见 `base.py:265` 到 `base.py:273`。
- FSDP2：保存 `dist_state/` 和 `trainer_state.pt`，见 `base.py:276` 到 `base.py:295`。

resume 时会校验：

- checkpoint 里的 `world_size` 是否等于当前 `world_size`。
- checkpoint 路径中的 iteration 是否与状态文件中的 iteration 一致。

校验逻辑在 `base.py:203` 到 `base.py:212`。这不是多余防御，主要是避免 optimizer shard、scheduler step、目录命名不一致导致 silent corruption。

## 4. 为什么 `BaseTrainer.train()` 不实现默认循环

`BaseTrainer.train()` 在 `lightx2v_train/lightx2v_train/trainers/base.py:297` 直接 `raise NotImplementedError`。

原因是不同训练算法差异很大：

- `flow`：一个模型，一个 loss，一个 optimizer。
- `dmd`：student / fake / teacher 三个模型，两个 optimizer，student 和 fake 交替更新。
- `video_dmd`：仍是 DMD 思路，但训练步调和采样窗口按视频 denoising schedule 改写。
- `video_ar_dmd`：还要处理 causal chunk、KV cache、cross-attention cache。
- `dopsd`：一个模型里挂 student/teacher 两套 LoRA，teacher 是 student EMA。

如果强行在基类抽象统一训练循环，会让基类变成复杂的半成品框架。当前设计更务实：公共基础设施放基类，算法流程留给子类。

## 5. `FlowMatchingTrainer`：最基础、最值得先理解

`FlowMatchingTrainer` 位于 `lightx2v_train/lightx2v_train/trainers/flow.py:13`。它是最接近“标准训练器”的实现。

### 5.1 loss 计算

`compute_loss_on_sample()` 的步骤在 `flow.py:16` 到 `flow.py:32`：

1. `model.encode_to_latent(sample)` 把训练样本编码为 latent。
2. 采样同 shape 噪声。
3. `noise_scheduler.sample_timestep_or_sigma()` 采样 timestep/sigma。
4. `noise_scheduler.add_noise()` 构造 noisy latent。
5. `model.encode_condition(sample)` 编码 prompt、图像条件等。
6. `model.prepare_denoiser_input()` 做模型特定输入整理。
7. `model.denoise()` 前向预测 velocity。
8. `model.postprocess_denoiser_output()` 转回统一格式。
9. `noise_scheduler.build_train_gt(latent, noise)` 构造目标 `noise - latent`。
10. 对 prediction 和 target 做 MSE。

这说明新增模型时，优先目标不是改 trainer，而是让模型适配器实现这些方法。

### 5.2 训练循环

`train()` 位于 `flow.py:34` 到 `flow.py:105`，逻辑是：

```text
resolve_resume
  -> setup
  -> 可选初始 inference
  -> while current_iter < max_train_iters:
       for sample in dataloader:
         控制梯度同步
         loss / grad_accum_iters backward
         梯度累积未满则 continue
         clip_grad_norm
         optimizer.step
         lr_scheduler.step
         optimizer.zero_grad
         日志
         定期 save_checkpoint
         定期 run_inference
```

这就是后续理解其他 trainer 的“母版”。

## 6. `TFTrainer`：复用 flow 循环，只改 loss

`TFTrainer` 位于 `lightx2v_train/lightx2v_train/trainers/tf.py:10`，继承 `FlowMatchingTrainer`。

它没有重写 `train()`，只重写 `compute_loss_on_sample()`，见 `tf.py:31` 到 `tf.py:75`。这说明它认为：

- 梯度累积、optimizer、checkpoint、inference 都和 flow 一样。
- 差异只在如何构造 latent、sigma、mask 和 loss。

它的约束比较强：

- 只支持 `training.train_type='full'`，见 `tf.py:15` 到 `tf.py:16`。
- 只实现 `teacher_forcing.mode='chunkwise'`，见 `tf.py:18` 到 `tf.py:21`。
- 当前要求模型有 `denoise_teacher_forcing()`，否则报错，见 `tf.py:31` 到 `tf.py:33`。

因此它不是通用 trainer，而是 Wan causal/teacher-forcing 场景的专项 trainer。

## 7. `DmdTrainer`：DMD 蒸馏的算法容器

`DmdTrainer` 位于 `lightx2v_train/lightx2v_train/trainers/dmd.py:25`。它比 flow 复杂很多，核心原因是它同时管理三套角色：

- `self.model`：student，需要被训练。
- `self.fake_model`：fake flow model，也需要被训练。
- `self.teacher_model`：teacher，只推理不训练。

### 7.1 setup 阶段

`DmdTrainer.setup()` 位于 `dmd.py:63` 到 `dmd.py:124`：

- 先调用 `super().setup(resume_ckpt_path=None)` 初始化 student。
- 从 `model.fake` 配置构造 `fake_model`。
- 从 `model.teacher` 配置构造 `teacher_model`。
- fake model 和 teacher model 都通过 `build_model()` 创建，说明它们也是普通模型适配器，而不是特殊内部对象。
- teacher 的 transformer 被 `requires_grad_(False)` 并设为 eval。
- fake 单独拥有 optimizer 和 lr scheduler。
- DMD 使用专门的 `DMDFlowMatchingScheduler`。

这个设计的好处是：student/fake/teacher 可以共享同一种模型适配接口，也可以通过配置指向不同权重路径或 dtype。

### 7.2 DMD loss 思路

关键函数：

- `_predict_velocity()`：统一调用模型 denoiser，见 `dmd.py:203` 到 `dmd.py:207`。
- `_predict_teacher_velocity()`：带 CFG 的 teacher 预测，见 `dmd.py:209` 到 `dmd.py:222`。
- `_dmd_loss()`：用 fake 与 teacher 的预测差构造 distillation gradient，见 `dmd.py:145` 到 `dmd.py:154`。
- `run_back_simulation()`：从噪声出发跑若干 denoising step，见 `dmd.py:236` 到 `dmd.py:252`。
- `forward_loss()`：根据 `stage` 计算 student 或 fake 的 loss，见 `dmd.py:282` 到 `dmd.py:317`。

直观理解：

- 训练 student：让 student 生成的样本在 teacher/fake 对比信号下被修正。
- 训练 fake：让 fake 学会对 student 生成样本做 flow matching。
- teacher 不训练，只提供高质量方向。

### 7.3 训练循环

`DmdTrainer.train()` 位于 `dmd.py:319` 到 `dmd.py:443`。它和 flow 最大区别是一个 iteration 内部有两个阶段：

1. 对 student 做 forward/backward，缓存当前 microbatch 的 latent shape 和 condition。
2. student optimizer step。
3. 对 fake 执行 `fake_update_ratio` 次更新，复用刚才的 microbatch 信息。
4. fake optimizer step。
5. 日志、checkpoint、inference。

这就是为什么 DMD 不适合硬塞进 `FlowMatchingTrainer.compute_loss_on_sample()`：它不只是 loss 不同，而是训练拓扑不同。

## 8. `VideoDmdTrainer` 与 `VideoArDmdTrainer`

`VideoDmdTrainer` 位于 `dmd.py:619`。它继承 `DmdTrainer`，但做了视频和 Wan 场景的限制：

- `model.name` 必须在 `{"wan_t2v"}`，见 `dmd.py:625` 到 `dmd.py:627`。
- 只支持 `train_type='full'`，见 `dmd.py:628` 到 `dmd.py:629`。
- 通过 `denoising_step_list` / `warp_denoising_step` 构造视频采样 schedule，见 `dmd.py:631` 到 `dmd.py:643`。

它重写的重点：

- `setup()`：可加载 student checkpoint，并重设 student/fake scheduler 步数，见 `dmd.py:652` 到 `dmd.py:673`。
- `train()`：按 `fake_update_ratio` 控制 student 是否更新，fake 每轮更新，见 `dmd.py:711` 到 `dmd.py:772`。
- `_latent_shape()`：根据视频帧数、VAE temporal/spatial scale 推出 latent shape，见 `dmd.py:829` 到 `dmd.py:849`。
- `run_back_simulation()`：用视频 denoising steps 回推，见 `dmd.py:890` 到 `dmd.py:914`。

`VideoArDmdTrainer` 位于 `dmd.py:949`，只允许 `wan_t2v_ar`，见 `dmd.py:950` 到 `dmd.py:951`。它的关键差异是自回归 causal chunk：

- 按 `num_frame_per_chunk` 切 latent，见 `dmd.py:975` 到 `dmd.py:982`。
- 为 transformer 准备 KV cache 和 cross-attention cache，见 `dmd.py:1088` 到 `dmd.py:1114`。
- 每个 chunk 内按 denoising step 前进，并在 exit step 保留梯度，见 `dmd.py:984` 到 `dmd.py:1008`。

这说明 `video_ar_dmd` 已经非常模型结构相关，不是通用 DMD。

## 9. `DopsdTrainer`：双 LoRA + EMA teacher

`DopsdTrainer` 位于 `lightx2v_train/lightx2v_train/trainers/dopsd.py:27`。它的设计和 DMD 不一样：不是创建三套模型，而是在同一个 denoiser 上挂两套 LoRA adapter：

- student adapter：参与训练。
- teacher adapter：由 student adapter EMA 更新。

setup 逻辑在 `dopsd.py:94` 到 `dopsd.py:132`：

- 调用 `model.add_dual_lora()` 添加 student/teacher 两套 adapter。
- 调用 `model.set_dual_lora_trainable()` 只让 student LoRA 可训练。
- 构造 optimizer / lr scheduler。
- resume 时恢复 student，并尽量恢复 teacher EMA LoRA。

loss 计算在 `dopsd.py:245` 到 `dopsd.py:321`：

1. 要求 sample 有 `target_image`。
2. student 用数据集 prompt 编码 condition。
3. teacher 用编辑 prompt 和 reference image 编码 condition。
4. 从初始 latent 开始跑多步。
5. 每步 teacher 预测 `x0_teacher`，student 预测 `x0_student`。
6. 用 `MSE(x0_student, x0_teacher)` 训练 student。

训练循环在 `dopsd.py:354` 到 `dopsd.py:466`：

- 正常梯度累积、clip、optimizer step。
- 每次 optimizer step 后调用 `model.ema_update_lora_adapter()` 更新 teacher，见 `dopsd.py:420` 到 `dopsd.py:425`。
- 可选保存 student/teacher trajectory 可视化，见 `dopsd.py:401` 到 `dopsd.py:453`。
- 定期 inference 时会先跑 student，再单独跑 teacher inference，见 `dopsd.py:580` 到 `dopsd.py:593`。

checkpoint 保存也不同：它同时保存 student LoRA 和 teacher EMA LoRA，见 `dopsd.py:595` 到 `dopsd.py:629`。

## 10. trainer 与 model 的接口契约

如果目标是“快速适配新模型”，最重要的是理解 trainer 期待 model 提供什么，而不是先改 trainer。

基础接口来自 `BaseModel`，位于 `lightx2v_train/lightx2v_train/model_zoo/base.py:20`：

| 接口 | trainer 如何使用 |
|---|---|
| `denoiser_module()` | 返回真正训练的 transformer/denoiser。 |
| `load_components()` | 入口脚本加载模型组件。 |
| `add_lora()` / `set_lora_trainable()` | LoRA 训练。 |
| `set_full_trainable()` | 全参训练。 |
| `trainable_parameters()` | 构建 optimizer。 |
| `enable_gradient_checkpointing()` | 降低显存。 |
| `is_fsdp2_wrapped()` / `fsdp2_state_module()` / `fsdp2_shard_plan()` | FSDP2 训练和保存。 |
| `save_lora_weights()` / `load_lora_weights_for_resume()` | LoRA checkpoint。 |
| `encode_to_latent()` | flow loss 中把样本转 latent。 |
| `encode_condition()` | flow loss 中把样本转条件。 |
| `prepare_denoiser_input()` | 处理模型特定输入格式。 |
| `denoise()` | 执行 denoiser forward。 |
| `postprocess_denoiser_output()` | 将模型输出转成统一 velocity 格式。 |

额外算法有额外接口：

- `TFTrainer` 需要 `denoise_teacher_forcing()`。
- `DmdTrainer` 图像场景需要 `dmd_latent_shape()`、`encode_prompt_condition()` 等模型方法。
- `VideoDmdTrainer` / `VideoArDmdTrainer` 强依赖 Wan 模型的方法和 transformer 形态。
- `DopsdTrainer` 需要 dual LoRA、`predict_velocity()`、reference image latent 等能力。

结论：新增模型时，优先复用 `flow` trainer。只有当训练算法本身变了，才新增 trainer。

## 11. 配置如何驱动 trainer

以 `lightx2v_train/configs/train/flow/qwen_image_lora.yaml` 为例：

- `model.name: qwen_image` 选择模型适配器。
- `data.train.name: image_dataset` 选择数据构建器。
- `scheduler` 控制 timestep/sigma 采样与 time shift。
- `training.method: flow` 选择 `FlowMatchingTrainer`。
- `training.train_type: lora` 选择 LoRA 训练。
- `training.lora` 控制 rank、alpha、target modules。
- `training.optimizer` 控制 AdamW。
- `inference.method: image_infer` 控制训练期推理验证。
- `resume.auto_resume: true` 控制自动恢复。

配置被 `lightx2v_train/lightx2v_train/runtime/config.py:7` 到 `config.py:14` 读取，并用 OmegaConf 解析 `${...}` 插值。

## 12. 设计优点和真实代价

### 优点

1. **入口干净**：`train.py` 只负责编排，不关心具体算法。
2. **算法可插拔**：`training.method` 通过 registry 选择 trainer。
3. **模型适配层清晰**：trainer 通过 `BaseModel` 方法调用模型，不直接依赖大多数模型内部细节。
4. **优化能力集中复用**：LoRA/full、DDP/FSDP2、gradient checkpointing、checkpoint、训练期 inference 在基类复用。
5. **复杂算法可独立表达**：DMD、D-OPSD 不被迫套进一个过度抽象的统一循环。

### 代价

1. **模块命名容易误解**：训练代码在 `lightx2v_train` 子项目，不在 `lightx2v.trainer`。
2. **接口契约偏隐式**：很多模型方法靠运行时报错，不是 Protocol/ABC 静态约束。
3. **DMD / DOPSD 有重复基础设施**：例如 DOPSD 重新实现了一些 base 中已有的配置和 checkpoint 逻辑。
4. **部分 trainer 模型绑定较强**：`teacher_forcing`、`video_dmd`、`video_ar_dmd` 明显是 Wan 专项。
5. **checkpoint 格式随算法变化**：flow、DMD、DOPSD 的权重组织不同，恢复和部署时要看具体 trainer。

## 13. 适配新模型的实践路线

如果我要在这个框架上适配新模型，我会按这个顺序做：

1. **先实现模型适配器**：在 `lightx2v_train/lightx2v_train/model_zoo` 新增模型类，注册到 `MODEL_REGISTER`。
2. **优先跑 `flow` LoRA**：只要实现 `encode_to_latent()`、`encode_condition()`、`prepare_denoiser_input()`、`denoise()`、`postprocess_denoiser_output()`，就能复用大部分训练基础设施。
3. **再考虑 full / FSDP2**：补 `fsdp2_shard_plan()`，确认 denoiser 在 `self.transformer` 上，符合 DDP/FSDP2 包装预期。
4. **最后才新增 trainer**：只有算法需要多模型、多 optimizer、EMA teacher、特殊 rollout 时，才写新的 trainer。
5. **不要暗猜 sample schema**：先看 dataset 输出字段，再看模型 `encode_*` 方法如何消费字段。

## 14. 你接下来很可能会问的问题

### Q1：我适配新模型时，到底该改 model 还是 trainer？

默认改 model。只要你的训练目标仍是 flow matching，就不应该新建 trainer。新模型应该实现 `BaseModel` 里的统一方法，让 `FlowMatchingTrainer` 直接可用。

### Q2：什么时候必须新建 trainer？

当训练循环本身变了才新建。例如：

- 需要 student / teacher / fake 多模型。
- 需要两个 optimizer 或多阶段交替更新。
- 需要 EMA teacher adapter。
- 需要 rollout / back simulation / cache-aware chunkwise 训练。

这类变化不是 `compute_loss_on_sample()` 能干净表达的。

### Q3：为什么 trainer 里会跑 inference？

因为训练期验证需要复用同一个模型状态和 checkpoint step。`BaseTrainer.run_inference()` 会把输出放到 `inference.output_dir/iter-xxxxxxxxx`，推理后再恢复 trainable 状态。这比外部脚本猜 checkpoint 路径更稳。

### Q4：为什么 resume 要检查 `world_size`？

普通 optimizer state 和 FSDP2 sharded state 都可能依赖训练时的 world size。如果用不同 world size 盲目恢复，可能直接报错，也可能更糟：状态错位但没有立刻报错。所以代码在 `base.py:203` 到 `base.py:212` 做硬校验。

### Q5：为什么 `apply_parallel()` 放在 LoRA/full trainable 设置之后？

因为 LoRA adapter 必须先挂到 denoiser 上，requires_grad 也要先明确。DDP/FSDP2 包装的是已经准备好的 denoiser；否则可能出现 adapter 没被包装、可训练参数不完整、或者 checkpoint state dict 不符合预期的问题。

### Q6：这个 trainer 模块的目的到底是什么？

它的目的不是“实现某个模型”，而是实现训练控制面。更具体地说，它负责把配置、模型适配器、数据、优化器、分布式、checkpoint、训练期 inference 和具体训练算法串起来。

可以把它理解成 LightX2V 训练子项目里的算法调度层：

- `train.py` 只负责启动流程，不关心具体训练算法。
- `training.method` 决定使用哪个 trainer。
- `BaseTrainer` 复用公共训练基础设施，例如 LoRA/full 参数选择、optimizer、lr scheduler、DDP/FSDP2 包装、resume、save、inference hook。
- `FlowMatchingTrainer`、`DmdTrainer`、`DopsdTrainer` 等子类负责不同训练算法的 loss 和训练拓扑。
- 模型内部细节留给 `model_zoo` 适配器，trainer 只通过统一方法调用模型。

所以它解决的核心问题是：**让新增模型尽量复用已有训练流程，让新增训练算法只改 trainer 层，而不是把所有逻辑塞进入口脚本或模型类里。**

### Q7：训练时依赖其他训练后端吗？

如果“训练后端”指 HuggingFace `Trainer`、Accelerate、DeepSpeed、PyTorch Lightning 这类高级训练框架，那么当前 `lightx2v_train.trainers` 没有依赖它们。代码里训练循环是 LightX2V 自己写的。

它实际依赖的是更底层的 PyTorch 训练能力：

- optimizer 用 `torch.optim.AdamW`。
- 单机多卡数据并行可用 PyTorch `DistributedDataParallel`。
- 全参大模型训练可用 PyTorch FSDP2，也就是 `torch.distributed.fsdp.fully_shard`。
- 分布式 checkpoint 用 `torch.distributed.checkpoint`。
- 学习率 scheduler 复用 `diffusers.optimization.get_scheduler`。
- LoRA 的 adapter 注入和权重保存主要在 `model_zoo.BaseModel` 中通过 PEFT / Diffusers 辅助完成。

如果是单卡或非分布式训练，`apply_parallel()` 会直接返回，不启用 DDP/FSDP2。也就是说，trainer 不强制要求外部训练后端；它按配置选择是否使用 PyTorch 原生分布式能力。

需要区分两类依赖：

- **训练控制后端**：没有依赖 DeepSpeed/Accelerate/Lightning 这类框架，训练 loop 自己实现。
- **模型组件后端**：具体模型可能依赖 Diffusers、Transformers、PEFT、原生 Wan 模块或自定义算子，但这些属于模型适配器和推理组件，不是 trainer 的控制后端。

实践判断标准很简单：如果你只是接新模型，优先实现 `model_zoo` 适配器并复用 `flow` trainer；如果你想接入新的训练后端，例如 DeepSpeed，那才需要改 `runtime/parallel.py`、checkpoint 逻辑和 trainer 的 gradient sync / state 保存约定。

## 15. 阅读源码建议顺序

建议按下面顺序读，不要从 DMD 直接开始：

1. `lightx2v_train/train.py`
2. `lightx2v_train/lightx2v_train/utils/registry.py`
3. `lightx2v_train/lightx2v_train/trainers/base.py`
4. `lightx2v_train/lightx2v_train/trainers/flow.py`
5. `lightx2v_train/lightx2v_train/model_zoo/base.py`
6. 一个具体模型，例如 `lightx2v_train/lightx2v_train/model_zoo/qwen_image.py`
7. 再读 `dmd.py` / `dopsd.py`

这样能先建立“训练器如何调用模型”的主线，再看复杂算法。
