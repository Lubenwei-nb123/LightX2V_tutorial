# 第6课：Offload 机制实现

## 学习目标
- 理解视频生成推理中显存压力的来源和分布
- 掌握三种 Offload 粒度（model / block / phase）的实现原理和适用场景
- 理解双缓冲 + CUDA Stream 异步传输的工作方式
- 掌握 Disk Offload（lazy_load）的三级流水线设计
- 具备为新模型配置和调试 Offload 的能力

---

## 0. 为什么需要 Offload

Wan 2.1 14B 模型推理时的显存占用估算（I2V 480p, 81帧, BF16, dim=5120, 40层）：

```
组件                      显存占用（BF16）  占比     常驻/临时
─────────────────────────────────────────────────────────
DiT 权重（40 Block）        ~60 GB         86%     常驻
  - 每个 Block              ~1.5 GB
  - 非 Block（norm/head）   ~0.9 GB
T5 编码器                   ~5 GB          7%      编码阶段常驻
CLIP 编码器                 ~1.5 GB        2%      编码阶段常驻
VAE                        ~0.5 GB         1%      编/解码阶段常驻
─────────────────────────────────────────────────────────
权重小计                    ~68 GB

激活值（DiT 前向）          ~3 GB                   每 step 临时
latent + 中间张量           ~1 GB                   常驻
─────────────────────────────────────────────────────────
```

**注意**：上面是纯 BF16 的数据。实际使用 Offload 时几乎总是搭配 FP8 量化（权重大小减半），所以后面的分析以 **FP8 + Offload** 为主。

以下是实测的 Block 内部各 Phase 大小（基于 `/data/phd/hf_models/Wan2.1-I2V-14B-480P`，原始权重为 FP32 存储）：

```
Phase                        参数量    FP32 存储（实测）  BF16 运行时   FP8 运行时
────────────────────────────────────────────────────────────────────────────
Phase 0 (SelfAttn+norm1+mod)  ~105M    400 MB           ~200 MB      ~100 MB
Phase 1 (CrossAttn+norm3)     ~157M    600 MB           ~300 MB      ~150 MB
Phase 2 (FFN+norm2)           ~142M    540 MB           ~270 MB      ~135 MB
────────────────────────────────────────────────────────────────────────────
整个 Block                    ~404M    1540 MB          ~770 MB      ~385 MB
```

**核心矛盾**：4090 只有 24GB 显存，BF16 下连 DiT 都放不下。但注意到一个关键事实——**T5 和 CLIP 只在编码阶段用一次，DiT 的 40 个 Block 在每个 step 中是顺序执行的**。这意味着它们不需要同时在 GPU 上。

Offload 的核心思想：**把当前不用的权重放到 CPU 内存（甚至磁盘）上，用到时再搬回 GPU**。CPU 内存通常有 64GB-256GB，足够放下所有权重。

---

## 1. 三种 Offload 粒度概览

```
┌─────────────────────────────────────────────────────────────┐
│  "model" 粒度：整个 DiT 为单位                                │
│  GPU 上：1 个完整 DiT（BF16 ~60GB / FP8 ~30GB）              │
│  适用：MoE 双模型切换                                         │
│  实现：MultiModelStruct 中的 to_cpu() / to_cuda()            │
├─────────────────────────────────────────────────────────────┤
│  "block" 粒度：单个 Transformer Block 为单位                  │
│  GPU 上：2 个 Block 的权重（双缓冲，FP8 ~770MB）               │
│  适用：24GB 显卡（如 4090）                                   │
│  实现：WeightAsyncStreamManager + infer_with_blocks_offload  │
├─────────────────────────────────────────────────────────────┤
│  "phase" 粒度：Block 内的子模块为单位                          │
│  GPU 上：1 个 Phase 的权重（FP8 最大 ~150MB）                  │
│  适用：16GB 显卡或更极端的显存限制                             │
│  实现：WeightAsyncStreamManager + infer_with_phases_offload  │
└─────────────────────────────────────────────────────────────┘
```

再叠加 Disk Offload（lazy_load）后，权重可以完全不占 CPU 内存：

```
               显存需求（DiT 权重部分，FP8 量化）
─────────────────────────────────────────────
无 Offload       ~30 GB（全部 Block 在 GPU）
block Offload    ~770 MB（2 个 FP8 Block 在 GPU）
phase Offload    ~150 MB（1 个最大 FP8 Phase 在 GPU）
lazy_load        同上，但 CPU 内存也省掉
```

---

## 2. 编码器的 Offload

在进入 DiT 的 block/phase Offload 之前，先看编码器的 Offload——它更简单但也很重要。

### 2.1 阶段级 Offload：编码器用完即卸载

第3课讲过，Runner 的 `run_input_encoder()` 中编码器是**顺序调用**的：

```
T5 编码 → CLIP 编码 → VAE 编码 → 释放编码器 → DiT 推理
```

这意味着编码器和 DiT 不需要同时在 GPU 上。两种实现方式：

**方式1：lazy_load（加载-使用-卸载）**

```python
# wan_runner.py:244-279（第3课讲过的模式）
def run_text_encoder(self, input_info):
    if self.config.get("lazy_load", False):
        self.text_encoders = self.load_text_encoder()   # 加载到 GPU
    context = self.text_encoders[0].infer([prompt])      # 使用
    if self.config.get("lazy_load", False):
        del self.text_encoders[0]                        # 卸载
        torch_device_module.empty_cache()
```

**方式2：cpu_offload（权重在 CPU，用时搬到 GPU）**

```python
# wan_runner.py:121-125
t5_offload = self.config.get("t5_cpu_offload", self.config.get("cpu_offload"))
if t5_offload:
    t5_device = torch.device("cpu")   # 权重加载到 CPU
text_encoder = T5EncoderModel(device=t5_device, cpu_offload=t5_offload, ...)
# 编码器内部的 infer() 会在计算前把权重搬到 GPU，计算后搬回 CPU
```

**两种方式的区别**：

| 维度 | lazy_load | cpu_offload |
|------|-----------|-------------|
| 首次使用 | 从磁盘加载到 GPU | 从 CPU 搬到 GPU |
| 速度 | 慢（涉及磁盘 IO） | 快（CPU↔GPU 传输） |
| CPU 内存 | 不占用 | 占用（权重常驻 CPU） |
| 适用场景 | CPU 内存也紧张 | CPU 内存充裕 |

### 2.2 WanModel.infer() 中的阶段切换

DiT 推理开始前，需要把 pre_weight（patch_embedding 等小权重）搬到 GPU：

```python
# wan/model.py:159-166
def infer(self, inputs):
    if self.cpu_offload:
        if self.offload_granularity != "model":
            self.pre_weight.to_cuda()                          # 搬小权重到 GPU
            self.transformer_weights.non_block_weights_to_cuda() # norm + head

    # ... DiT 推理 ...

    if self.cpu_offload:
        if self.offload_granularity != "model":
            self.pre_weight.to_cpu()                           # 搬回 CPU
            self.transformer_weights.non_block_weights_to_cpu()
```

**注意**：Block 权重不在这里搬——它们由 `offload_manager` 在推理循环内逐个管理。

### 2.3 无 Offload 时的完整调用链

作为对比基准，先看无 Offload 时的调用链：

```
Pipeline.generate()
 └── Runner.run_pipeline(input_info)
      │
      ├── run_input_encoder()
      │     ├── T5.infer(prompt)          ← T5 权重在 GPU，直接算
      │     ├── CLIP.visual(image)        ← CLIP 权重在 GPU，直接算
      │     └── VAE.encode(image)         ← VAE 权重在 GPU，直接算
      │
      └── run_main()
           ├── init_run()
           │     └── scheduler.prepare()  ← 生成初始噪声
           │
           ├── for step in range(infer_steps):
           │     ├── scheduler.step_pre()
           │     │
           │     ├── model.infer(inputs)
           │     │     │ (cpu_offload=False，不做任何搬运)
           │     │     │
           │     │     └── _infer_cond_uncond()
           │     │           ├── pre_infer.infer(pre_weight, ...)
           │     │           │     └── pre_weight 在 GPU，直接用
           │     │           │
           │     │           ├── transformer_infer.infer(transformer_weights, ...)
           │     │           │     └── infer_without_offload():
           │     │           │           for block_idx in range(40):
           │     │           │               infer_block(blocks[block_idx], ...)
           │     │           │               └── 所有 40 个 Block 都在 GPU
           │     │           │
           │     │           └── post_infer.infer(x, ...)
           │     │
           │     └── scheduler.step_post()
           │
           └── VAE.decode(latents)        ← VAE 权重在 GPU，直接算
```

**GPU 显存峰值**：BF16 下 ~65GB（所有权重 + 激活值 + latent 同时在 GPU），远超任何消费级显卡

---

## 3. Block 级 Offload：双缓冲详解

### 3.1 核心思想

GPU 上只保留 **2 个 Block 大小的缓冲区**（双缓冲）。遍历 40 个 Block 时，当前 Block 在 buffer[0] 上计算，同时异步地把下一个 Block 的权重从 CPU 传到 buffer[1]。计算完成后交换两个 buffer 的角色。

```
时间 →
GPU buffer[0]: [Block 0 计算]              [Block 2 计算]
GPU buffer[1]:              [Block 1 计算]              [Block 3 计算]
CPU→GPU:       [Block 1 传输][Block 2 传输][Block 3 传输][Block 4 传输]
               ↑ 在 cuda_load_stream 上    ↑ 与计算重叠
```

### 3.2 WeightModule 和 WeightModuleList

在看缓冲区创建代码之前，先了解权重容器的基类。框架没有使用 PyTorch 的 `nn.Module`，而是自己实现了一套轻量的权重管理体系：

```python
# lightx2v/common/modules/weight_module.py:4-27
class WeightModule:
    """所有权重容器的基类"""
    def __init__(self):
        self._modules = {}      # 子模块（嵌套的 WeightModule）
        self._parameters = {}   # 直接持有的权重

    def add_module(self, name, module):
        self._modules[name] = module
        setattr(self, name, module)   # 可以通过 self.xxx 访问

    def load(self, weight_dict):
        """从权重字典中加载所有子模块和参数的权重"""
        for _, module in self._modules.items():
            if hasattr(module, "load"):
                module.load(weight_dict)    # 递归加载
        for _, parameter in self._parameters.items():
            if hasattr(parameter, "load"):
                parameter.load(weight_dict)

    def to_cpu(self, non_blocking=False):
        """把所有权重搬到 CPU（递归处理子模块）"""
        for name, param in self._parameters.items():
            if param is not None and hasattr(param, "to_cpu"):
                param.to_cpu()
        for module in self._modules.values():
            if isinstance(module, WeightModuleList):
                for i in range(len(module)):
                    for m in module[i]._modules.values():
                        if m is not None and hasattr(m, "to_cpu"):
                            m.to_cpu()
            else:
                if module is not None and hasattr(module, "to_cpu"):
                    module.to_cpu()

    def to_cuda(self, non_blocking=False):
        """把所有权重搬到 GPU（递归处理子模块）"""
        # 结构同 to_cpu，方向相反

    def state_dict(self, destination=None):
        """导出所有权重为字典（用于 Offload 时的 buffer 间拷贝）"""
        ...

    def load_state_dict(self, destination, block_index, adapter_block_index=None):
        """从字典加载权重到当前容器（Offload buffer 间拷贝的核心方法）"""
        ...

    def load_state_dict_from_disk(self, block_index, adapter_block_index=None):
        """从磁盘加载权重（lazy_load 用）"""
        ...
```

`WeightModuleList` 是 `WeightModule` 的列表版本，类似 PyTorch 的 `nn.ModuleList`：

```python
# lightx2v/common/modules/weight_module.py:191-216
class WeightModuleList(WeightModule):
    """可索引的 WeightModule 列表"""
    def __init__(self, modules=None):
        super().__init__()
        self._list = []
        if modules is not None:
            for idx, module in enumerate(modules):
                self.append(module)

    def append(self, module):
        idx = len(self._list)
        self._list.append(module)
        self.add_module(str(idx), module)  # 用字符串索引注册为子模块

    def __getitem__(self, idx):
        return self._list[idx]

    def __setitem__(self, idx, module):
        self._list[idx] = module
        self.add_module(str(idx), module)

    def __len__(self):
        return len(self._list)

    def __iter__(self):
        return iter(self._list)
```

**为什么不用 PyTorch 的 nn.Module？**

因为推理框架不需要 `nn.Module` 的大部分功能（自动求导、参数注册到优化器、hook 机制等），但需要一些 `nn.Module` 不原生支持的能力：
- `to_cpu()` / `to_cuda()`：递归地搬运所有权重，包括量化权重（FP8 tensor 不是标准的 `nn.Parameter`）
- `state_dict()` / `load_state_dict()`：支持按 `block_index` 参数化加载（Offload 双缓冲需要把不同 Block 的权重加载到同一个 buffer 中）
- `load_state_dict_from_disk()`：支持按需从磁盘读取

**在 Offload 中的角色**：

```
WanTransformerWeights (WeightModule)
    ├── blocks (WeightModuleList)           ← 40 个 Block 的实际权重
    │     ├── [0] WanTransformerAttentionBlock (WeightModule)
    │     ├── [1] ...
    │     └── [39] ...
    │
    ├── offload_block_cuda_buffers (WeightModuleList)  ← GPU 双缓冲
    │     ├── [0] WanTransformerAttentionBlock (WeightModule, create_cuda_buffer=True)
    │     └── [1] WanTransformerAttentionBlock (WeightModule, create_cuda_buffer=True)
    │
    └── norm, head, head_modulation          ← 非 Block 权重（小，直接 to_cuda/to_cpu）
```

Offload 的核心操作就是调用 `cuda_buffers[1].load_state_dict(blocks[N].state_dict(), N)`——把第 N 个 Block 的权重拷贝到 GPU buffer 中。

### 3.3 缓冲区的创建

```python
# wan/weights/transformer_weights.py:55-97
def register_offload_buffers(self, config, lazy_load_path, lora_path):
    if config["cpu_offload"]:
        if config["offload_granularity"] == "block":
            self.offload_blocks_num = 2   # 双缓冲

            # GPU 上创建 2 个 Block 大小的空壳（有相同结构但权重为空）
            self.offload_block_cuda_buffers = WeightModuleList([
                WanTransformerAttentionBlock(
                    block_index=i,
                    create_cuda_buffer=True,   # ← GPU buffer
                    create_cpu_buffer=False,
                    ...
                )
                for i in range(self.offload_blocks_num)  # 2 个
            ])
```

`create_cuda_buffer=True` 的含义：创建一个和真实 Block 结构完全一样的权重容器，但**权重张量分配在 GPU 上且初始为空**，用于接收从 CPU 传输过来的权重。

### 3.4 推理循环：infer_with_blocks_offload

```python
# wan/infer/offload/transformer_infer.py:39-71
def infer_with_blocks_offload(self, blocks, x, pre_infer_out):
    for block_idx in range(len(blocks)):
        self.block_idx = block_idx

        # lazy_load 模式：提前从磁盘预取下一个 Block 到 CPU
        if self.lazy_load:
            next_prefetch = (block_idx + 1) % len(blocks)
            self.offload_manager.start_prefetch_block(next_prefetch)

        # 第一次：把 Block 0 的权重加载到 cuda_buffers[0]
        if self.offload_manager.need_init_first_buffer:
            self.offload_manager.init_first_buffer(blocks)

        # lazy_load 模式：等待磁盘预取完成，交换 CPU buffer
        if self.lazy_load:
            self.offload_manager.swap_cpu_buffers()

        # 异步传输：在 cuda_load_stream 上把下一个 Block 传到 cuda_buffers[1]
        self.offload_manager.prefetch_weights(
            (block_idx + 1) % len(blocks), blocks
        )

        # 计算：在 compute_stream 上用 cuda_buffers[0] 的权重推理当前 Block
        with torch_device_module.stream(self.offload_manager.compute_stream):
            x = self.infer_block(self.offload_manager.cuda_buffers[0], x, pre_infer_out)

        # 同步 + 交换：等传输和计算都完成，交换两个 buffer 的角色
        self.offload_manager.swap_blocks()

    return x
```

### 3.5 CUDA Stream 的关键作用

双缓冲的异步性依赖于 **CUDA Stream**——GPU 上的独立执行队列：

```python
# common/offload/manager.py:14-26
class WeightAsyncStreamManager(object):
    def __init__(self, offload_granularity):
        self.init_stream = torch_device_module.Stream(priority=0)
        # 两个独立的 stream：传输和计算可以并行
        self.cuda_load_stream = torch_device_module.Stream(priority=1)   # 传输用
        self.compute_stream = torch_device_module.Stream(priority=-1)    # 计算用
```

**为什么需要两个 Stream？**

默认情况下，GPU 上的所有操作都在 default stream 上**串行执行**。要让传输和计算并行，必须把它们放到不同的 stream 上：

```
default stream:  [op1] [op2] [op3] ...  ← 所有操作排队，无并行

两个 stream：
  cuda_load_stream:   [传输 Block N+1]          [传输 Block N+2]
  compute_stream:     [计算 Block N]             [计算 Block N+1]
                      ↑ 传输和计算在不同 stream 上，可以并行执行
```

**swap_blocks() 的同步逻辑**：

```python
# common/offload/manager.py:91-100
def swap_blocks(self):
    self.cuda_load_stream.synchronize()   # 等待传输完成
    self.compute_stream.synchronize()      # 等待计算完成
    # 两者都完成后，交换 buffer 角色
    self.cuda_buffers[0], self.cuda_buffers[1] = (
        self.cuda_buffers[1],
        self.cuda_buffers[0],
    )
```

必须同时等待两个 stream，因为：
- 如果传输没完就开始用 buffer[1] 计算 → 读到不完整的权重 → 计算结果错误
- 如果计算没完就往 buffer[0] 写入新权重 → 覆盖正在使用的权重 → 计算结果错误

### 3.6 prefetch_weights：异步传输的实现

```python
# common/offload/manager.py:77-82
def prefetch_weights(self, block_idx, blocks, adapter_block_idx=None):
    with torch_device_module.stream(self.cuda_load_stream):  # 在传输 stream 上执行
        if hasattr(self, "cpu_buffers"):
            # lazy_load 模式：从 CPU pinned memory 传到 GPU buffer
            self.cuda_buffers[1].load_state_dict(
                self.cpu_buffers[0].state_dict(), block_idx, adapter_block_idx
            )
        else:
            # 普通模式：直接从 CPU 上的 Block 权重传到 GPU buffer
            self.cuda_buffers[1].load_state_dict(
                blocks[block_idx].state_dict(), block_idx, adapter_block_idx
            )
```

`load_state_dict` 的本质是 `gpu_tensor.copy_(cpu_tensor)`——在 `cuda_load_stream` 上执行的异步 CPU→GPU 拷贝。

### 3.7 Block 级 Offload 的完整调用链

```
Pipeline.generate()
 └── Runner.run_pipeline(input_info)
      │
      ├── run_input_encoder()
      │     ├── T5.infer(prompt)          ← t5_cpu_offload 决定设备
      │     ├── CLIP.visual(image)
      │     └── VAE.encode(image)
      │     └── gc.collect() + empty_cache()  ← 释放编码器显存
      │
      └── run_main()
           ├── init_run()
           │     ├── [lazy_load] model = load_transformer()  ← 此时才加载 DiT
           │     └── scheduler.prepare()
           │
           ├── for step in range(infer_steps):
           │     ├── scheduler.step_pre()
           │     │
           │     ├── model.infer(inputs)
           │     │     │
           │     │     ├── pre_weight.to_cuda()                  ← ① 小权重搬到 GPU
           │     │     ├── transformer_weights.non_block_weights_to_cuda()
           │     │     │
           │     │     ├── _infer_cond_uncond()
           │     │     │     ├── pre_infer.infer(pre_weight, ...)
           │     │     │     │
           │     │     │     ├── transformer_infer.infer(transformer_weights, ...)
           │     │     │     │     └── infer_with_blocks_offload():  ← ② Block 循环
           │     │     │     │           │
           │     │     │     │           ├── init_first_buffer()    ← 首次：Block 0 → GPU buffer[0]
           │     │     │     │           │
           │     │     │     │           └── for block_idx in range(40):
           │     │     │     │                 │
           │     │     │     │                 ├── prefetch_weights(block_idx+1)
           │     │     │     │                 │     └── cuda_load_stream: blocks[N+1] → buffer[1]
           │     │     │     │                 │                            (异步，不阻塞)
           │     │     │     │                 │
           │     │     │     │                 ├── compute_stream:
           │     │     │     │                 │     infer_block(buffer[0], x, ...)
           │     │     │     │                 │     └── SelfAttn → CrossAttn → FFN
           │     │     │     │                 │         (用 buffer[0] 的权重计算)
           │     │     │     │                 │
           │     │     │     │                 └── swap_blocks()
           │     │     │     │                       ├── cuda_load_stream.synchronize()
           │     │     │     │                       ├── compute_stream.synchronize()
           │     │     │     │                       └── buffer[0], buffer[1] = buffer[1], buffer[0]
           │     │     │     │
           │     │     │     └── post_infer.infer(x, ...)
           │     │     │
           │     │     ├── pre_weight.to_cpu()                   ← ③ 小权重搬回 CPU
           │     │     └── transformer_weights.non_block_weights_to_cpu()
           │     │
           │     └── scheduler.step_post()
           │
           └── VAE.decode(latents)
```

**GPU 显存峰值**：~5GB（2 个 FP8 Block ~770MB + 小权重 ~100MB + 激活 ~3GB + latent ~1GB）

---

## 4. Phase 级 Offload：更细粒度

### 4.1 Phase 的划分

第3课修正过：phase 是 **Block 内部的子模块**。每个 Block 被拆成 3 个 phase：

```
WanTransformerAttentionBlock
    └── compute_phases = [
            WanSelfAttention,     # phase 0：BF16 ~400MB / FP8 ~200MB
            WanCrossAttention,    # phase 1：BF16 ~600MB / FP8 ~300MB
            WanFFN,               # phase 2：BF16 ~540MB / FP8 ~270MB
        ]
```

Phase 级 Offload 的 GPU 上同时只有 **1 个 Phase 的权重**（FP8 最大约 150MB），比 Block 级（FP8 约 385MB×2 ≈ 770MB）省很多。

同时也更新 Phase 大小注释（前面实测表格中的 FP32 存储值 ÷ 4 才是 FP8 运行时大小）：

```
WanTransformerAttentionBlock
    └── compute_phases = [
            WanSelfAttention,     # phase 0：FP8 ~100MB
            WanCrossAttention,    # phase 1：FP8 ~150MB
            WanFFN,               # phase 2：FP8 ~135MB
        ]
```

### 4.2 为什么三个 Phase 的大小接近？

这不是 Wan 模型的巧合，而是 **Transformer 架构的固有特性**。用参数量公式推导（d=dim, f=ffn_dim, 忽略 bias 和 norm 的小量）：

```
Phase 0 (SelfAttention):
  Q/K/V/O 四个投影，每个 d×d
  合计 ≈ 4d²

Phase 1 (CrossAttention):
  Text cross-attn: Q/K/V/O 四个投影 → 4d²
  Image cross-attn（I2V 才有）: K_img/V_img 两个投影 → 2d²
  合计 ≈ 4d²（T2V）或 6d²（I2V）

Phase 2 (FFN):
  up 投影: d×f + down 投影: f×d
  合计 ≈ 2df
```

三者的比值取决于 **d 和 f 的关系**：

```
SelfAttn : CrossAttn(I2V) : FFN = 4d² : 6d² : 2df

Wan 2.1: d=5120, f=13824
  4d² = 4 × 5120² = 104.9M
  6d² = 6 × 5120² = 157.3M
  2df = 2 × 5120 × 13824 = 141.6M

理论比值 → 105M : 157M : 142M
实测比值 → 105M : 157M : 142M  ✓ 完全吻合
```

大多数 Transformer 架构中 `f ≈ 2.5d ~ 4d`（Wan 是 2.7d），所以三个 phase 的参数量**天然在同一个数量级**。这是 phase 级 Offload 能有效工作的结构基础——如果某个 phase 占了 90% 参数，拆分就没有意义了。

### 4.3 为什么不把 SelfAttn 和 CrossAttn 合成一个 Phase？

既然 SelfAttn（4d²）和 CrossAttn（6d²）的大小接近，一个自然的问题：为什么不合成 2 个 phase 而是拆成 3 个？

```
方案 A（当前，3 phase）：
  Phase 0: SelfAttn        ~100 MB (FP8)
  Phase 1: CrossAttn       ~150 MB (FP8)
  Phase 2: FFN             ~135 MB (FP8)
  每个 Block 传输 3 次，最大 phase = 150 MB

方案 B（合并，2 phase）：
  Phase 0: SelfAttn+CrossAttn  ~250 MB (FP8)
  Phase 1: FFN                 ~135 MB (FP8)
  每个 Block 传输 2 次，最大 phase = 250 MB
```

| 维度 | 3 phase（当前） | 2 phase（合并） |
|------|----------------|----------------|
| GPU 权重峰值 | **150 MB** ← 更省 | 250 MB |
| 每 Block 传输次数 | 3 次 | **2 次** ← 更少 |
| 每 phase 计算时间 | 短 | **长** ← 更容易掩盖传输 |
| 传输与计算重叠效果 | 较差（计算太短，传输可能来不及） | **较好** |

合并方案的**吞吐量可能更高**——每个 phase 的计算量更大，传输更容易被计算掩盖（回顾第4课讨论：计算时间越长，双缓冲越有效）。但当前 3 phase 的设计优先考虑的是**极限显存节省**（150MB vs 250MB），因为 phase 级 Offload 的目标用户本身就是**极端显存受限**的场景（如 16GB 显卡），此时省 100MB 可能就是能跑和不能跑的区别。

### 4.4 缓冲区的创建

```python
# wan/weights/transformer_weights.py:99-111
elif config["offload_granularity"] == "phase":
    # 取一个 Block 的 compute_phases 作为 GPU buffer
    self.offload_phase_cuda_buffers = WanTransformerAttentionBlock(
        block_index=0,
        create_cuda_buffer=True,
        ...
    ).compute_phases   # ← 只取 phases 部分，不是完整 Block

    self.offload_block_cuda_buffers = None  # Block buffer 置空
```

**和 Block 级的关键区别**：
- Block 级：GPU 上有 **2 个 Block buffer**（双缓冲交替使用）
- Phase 级：GPU 上有 **3 个 Phase buffer**（对应 SelfAttn/CrossAttn/FFN，轮转使用）

### 4.5 推理循环：infer_with_phases_offload

```python
# wan/infer/offload/transformer_infer.py:73-114
def infer_with_phases_offload(self, blocks, x, pre_infer_out):
    for block_idx in range(len(blocks)):
        self.block_idx = block_idx
        if self.lazy_load:
            next_prefetch = (block_idx + 1) % len(blocks)
            self.offload_manager.start_prefetch_block(next_prefetch)

        # 在一个 Block 内部遍历 3 个 phase
        x = self.infer_phases(block_idx, blocks, x, pre_infer_out)
    return x

def infer_phases(self, block_idx, blocks, x, pre_infer_out):
    for phase_idx in range(self.phases_num):  # phases_num = 3
        if self.offload_manager.need_init_first_buffer:
            self.offload_manager.init_first_buffer(blocks)

        # 计算下一个要预取的 phase
        # 如果当前是最后一个 phase → 预取下一个 Block 的第一个 phase
        # 否则 → 预取当前 Block 的下一个 phase
        next_block_idx = (block_idx + 1) % len(blocks) \
            if phase_idx == self.phases_num - 1 else block_idx
        next_phase_idx = (phase_idx + 1) % self.phases_num

        if self.lazy_load:
            if phase_idx == self.phases_num - 1:
                self.offload_manager.swap_cpu_buffers()

        # 异步预取下一个 phase
        self.offload_manager.prefetch_phase(
            next_block_idx, next_phase_idx, blocks
        )

        # 计算当前 phase
        with torch_device_module.stream(self.offload_manager.compute_stream):
            x = self.infer_phase(
                phase_idx,
                self.offload_manager.cuda_buffers[phase_idx],
                x, pre_infer_out
            )

        self.offload_manager.swap_phases()
    return x
```

### 4.6 infer_phase()：phase 内部的计算

因为一个 Block 被拆成 3 个 phase，原本 `infer_block()` 中的连续计算被打断了。中间结果（如 SelfAttn 的输出 `y_out`）需要跨 phase 保存：

```python
# wan/infer/offload/transformer_infer.py:116-155
def infer_phase(self, cur_phase_idx, cur_phase, x, pre_infer_out):
    if cur_phase_idx == 0:
        # Phase 0: SelfAttention
        # 计算调制参数（存到 phase_params 供后续 phase 使用）
        (self.phase_params["shift_msa"], self.phase_params["scale_msa"],
         self.phase_params["gate_msa"], self.phase_params["c_shift_msa"],
         self.phase_params["c_scale_msa"], self.phase_params["c_gate_msa"],
        ) = self.pre_process(cur_phase.modulation, pre_infer_out.embed0)
        # SelfAttn 计算（输出存到 phase_params["y_out"]）
        self.phase_params["y_out"] = self.infer_self_attn(
            cur_phase, x, self.phase_params["shift_msa"], self.phase_params["scale_msa"]
        )

    elif cur_phase_idx == 1:
        # Phase 1: CrossAttention（需要 phase 0 的 y_out 和 gate_msa）
        x, self.phase_params["attn_out"] = self.infer_cross_attn(
            cur_phase, x, pre_infer_out.context,
            self.phase_params["y_out"], self.phase_params["gate_msa"]
        )

    elif cur_phase_idx == 2:
        # Phase 2: FFN（需要 phase 1 的 attn_out）
        self.phase_params["y"] = self.infer_ffn(
            cur_phase, x, self.phase_params["attn_out"],
            self.phase_params["c_shift_msa"], self.phase_params["c_scale_msa"]
        )
        x = self.post_process(
            x, self.phase_params["y"], self.phase_params["c_gate_msa"], pre_infer_out
        )
    return x
```

**`self.phase_params` 是 phase 级 Offload 特有的设计**——它用一个字典在 phase 之间传递中间结果。Block 级 Offload 不需要它，因为 `infer_block()` 中三个阶段是连续执行的，中间结果就是局部变量。

### 4.7 Block 级 vs Phase 级的对比

```
Block 级 Offload（一个 step 的时序）：
┌──────────────────────────────────────────────────────────┐
│ Block 0:                                                  │
│   GPU buffer[0]: [SelfAttn → CrossAttn → FFN]  ← 连续执行│
│   GPU buffer[1]: [传输 Block 1 权重]           ← 异步传输│
│ swap                                                      │
│ Block 1:                                                  │
│   GPU buffer[0]: [SelfAttn → CrossAttn → FFN]             │
│   GPU buffer[1]: [传输 Block 2 权重]                      │
│ ...                                                       │
└──────────────────────────────────────────────────────────┘
GPU 权重峰值：2 个 FP8 Block ≈ 770MB

Phase 级 Offload（一个 step 的时序）：
┌──────────────────────────────────────────────────────────┐
│ Block 0:                                                  │
│   Phase 0: [SelfAttn]     传输 Phase 1                   │
│   Phase 1: [CrossAttn]    传输 Phase 2                   │
│   Phase 2: [FFN]          传输 Block 1 Phase 0           │
│ Block 1:                                                  │
│   Phase 0: [SelfAttn]     传输 Phase 1                   │
│   ...                                                     │
└──────────────────────────────────────────────────────────┘
GPU 权重峰值：1 个最大 FP8 Phase ≈ 150MB
传输次数：40 × 3 = 120 次 vs Block 级的 40 次
```

**Phase 级省显存但慢**——传输次数多 3 倍，而且每个 phase 的计算量更小，传输能否被计算掩盖取决于具体硬件。

### 4.8 Phase 级 Offload 的完整调用链

```
Pipeline.generate()
 └── Runner.run_pipeline(input_info)
      │
      ├── run_input_encoder()  ← 同 Block 级
      │
      └── run_main()
           │
           ├── for step in range(infer_steps):
           │     │
           │     ├── model.infer(inputs)
           │     │     ├── pre_weight.to_cuda()
           │     │     │
           │     │     ├── _infer_cond_uncond()
           │     │     │     └── transformer_infer.infer()
           │     │     │           └── infer_with_phases_offload():
           │     │     │                 │
           │     │     │                 └── for block_idx in range(40):
           │     │     │                       └── infer_phases(block_idx):
           │     │     │                             │
           │     │     │                             ├── phase_idx=0 (SelfAttention):
           │     │     │                             │     ├── prefetch_phase(block, 1)
           │     │     │                             │     │     └── cuda_load_stream:
           │     │     │                             │     │         CrossAttn 权重 → buffer[1]
           │     │     │                             │     ├── compute_stream:
           │     │     │                             │     │     infer_phase(0, buffer[0], ...)
           │     │     │                             │     │     └── pre_process + infer_self_attn
           │     │     │                             │     │         结果存入 phase_params["y_out"]
           │     │     │                             │     └── swap_phases()
           │     │     │                             │
           │     │     │                             ├── phase_idx=1 (CrossAttention):
           │     │     │                             │     ├── prefetch_phase(block, 2)
           │     │     │                             │     │     └── FFN 权重 → buffer[2]
           │     │     │                             │     ├── compute_stream:
           │     │     │                             │     │     infer_phase(1, buffer[1], ...)
           │     │     │                             │     │     └── infer_cross_attn
           │     │     │                             │     │         读 phase_params["y_out"]
           │     │     │                             │     │         写 phase_params["attn_out"]
           │     │     │                             │     └── swap_phases()
           │     │     │                             │
           │     │     │                             └── phase_idx=2 (FFN):
           │     │     │                                   ├── prefetch_phase(block+1, 0)
           │     │     │                                   │     └── 下一个 Block 的 SelfAttn → buffer[0]
           │     │     │                                   ├── compute_stream:
           │     │     │                                   │     infer_phase(2, buffer[2], ...)
           │     │     │                                   │     └── infer_ffn + post_process
           │     │     │                                   │         读 phase_params["attn_out"]
           │     │     │                                   └── swap_phases()
           │     │     │
           │     │     └── pre_weight.to_cpu()
           │     │
           │     └── scheduler.step_post()
```

**GPU 显存峰值**：~4.2GB（1 个最大 FP8 Phase ~150MB + 激活 ~3GB + latent ~1GB）

---

## 5. Disk Offload（lazy_load）：三级流水线

当 CPU 内存也不够时，权重可以存在磁盘上。lazy_load 在 block/phase Offload 的基础上增加了 **磁盘→CPU** 的预取层。

### 5.1 三级存储层次

```
磁盘（SSD/NVMe）  →  CPU 内存（pinned memory）  →  GPU 显存
     block_0.safetensors        cpu_buffers              cuda_buffers
     block_1.safetensors
     ...
     block_39.safetensors
```

### 5.2 文件组织

lazy_load 要求权重按 Block 分文件存储（非 lazy_load 模式下所有权重在一个大文件里）：

```
model_path/
├── non_block.safetensors      ← pre_weight + post_weight（启动时加载）
├── block_0.safetensors        ← Block 0 权重（按需从磁盘读取）
├── block_1.safetensors
├── ...
└── block_47.safetensors
```

`_init_weights()` 时只加载 `non_block.safetensors`，Block 权重由 offload_manager 按需加载。

### 5.3 三级流水线

```
时间 →
磁盘→CPU:   [读 Block 2]         [读 Block 3]         [读 Block 4]
CPU→GPU:         [传 Block 1]         [传 Block 2]         [传 Block 3]
GPU 计算:             [计算 Block 0]        [计算 Block 1]        [计算 Block 2]
            ↑ ThreadPoolExecutor   ↑ cuda_load_stream   ↑ compute_stream
              异步磁盘 IO            异步 DMA              GPU 计算
```

三个操作分别在不同的执行上下文中并行：
- **磁盘→CPU**：`ThreadPoolExecutor` 的后台线程
- **CPU→GPU**：`cuda_load_stream` 上的异步 DMA
- **GPU 计算**：`compute_stream` 上的 GPU kernel

### 5.4 lazy_load 的初始化

```python
# common/offload/manager.py:124-128
def init_lazy_load(self, num_workers=6):
    self.lazy_load = True
    self.executor = ThreadPoolExecutor(max_workers=num_workers)  # 后台线程池
    self.prefetch_futures = []
    self.prefetch_block_idx = -1
```

### 5.5 磁盘预取流程

```python
# common/offload/manager.py:130-139
def start_prefetch_block(self, block_idx, adapter_block_idx=None):
    """在后台线程中从磁盘读取下一个 Block 到 CPU buffer"""
    self.prefetch_block_idx = block_idx
    self.prefetch_futures = []
    if self.offload_granularity == "block":
        # 提交一个任务：把 block_idx 的权重从磁盘读到 cpu_buffers[1]
        future = self.executor.submit(
            self.cpu_buffers[1].load_state_dict_from_disk, block_idx, adapter_block_idx
        )
        self.prefetch_futures.append(future)
    else:
        # phase 粒度：每个 phase 一个任务（可并行读取）
        for phase in self.cpu_buffers[1]:
            future = self.executor.submit(
                phase.load_state_dict_from_disk, block_idx, adapter_block_idx
            )
            self.prefetch_futures.append(future)
```

```python
# common/offload/manager.py:141-149
def swap_cpu_buffers(self):
    """等待磁盘预取完成，交换 CPU 双缓冲"""
    for f in self.prefetch_futures:
        f.result()  # 阻塞等待所有预取任务完成
    # 交换：cpu_buffers[1]（刚读好的）变成 [0]（下次传给 GPU 的）
    self.cpu_buffers = [self.cpu_buffers[1], self.cpu_buffers[0]]
```

### 5.6 lazy_load 在推理循环中的调用时序

```python
# wan/infer/offload/transformer_infer.py:39-71（标注 lazy_load 部分）
def infer_with_blocks_offload(self, blocks, x, pre_infer_out):
    for block_idx in range(len(blocks)):
        # ① 磁盘→CPU：后台线程预取下一个 Block
        if self.lazy_load:
            next_prefetch = (block_idx + 1) % len(blocks)
            self.offload_manager.start_prefetch_block(next_prefetch)

        if self.offload_manager.need_init_first_buffer:
            self.offload_manager.init_first_buffer(blocks)

        # ② 等待磁盘预取完成，交换 CPU buffer
        if self.lazy_load:
            self.offload_manager.swap_cpu_buffers()

        # ③ CPU→GPU：异步传输下一个 Block 到 GPU buffer[1]
        self.offload_manager.prefetch_weights((block_idx + 1) % len(blocks), blocks)

        # ④ GPU 计算：用 GPU buffer[0] 的权重推理当前 Block
        with torch_device_module.stream(self.offload_manager.compute_stream):
            x = self.infer_block(self.offload_manager.cuda_buffers[0], x, pre_infer_out)

        # ⑤ 等待传输和计算都完成，交换 GPU buffer
        self.offload_manager.swap_blocks()

    return x
```

### 5.7 warm_up_cpu_buffers：预热

首次推理前需要预热 CPU buffer，让磁盘预取的数据路径"热"起来：

```python
# common/offload/manager.py:109-122
def warm_up_cpu_buffers(self, blocks_num):
    logger.info("🔥 Warming up cpu buffers...")
    for i in tqdm(range(blocks_num)):
        for phase in self.cpu_buffers[0]:
            phase.load_state_dict_from_disk(i, None)
        for phase in self.cpu_buffers[1]:
            phase.load_state_dict_from_disk(i, None)
    # 预热完毕后，重新加载 Block 0 和 Block 1 到两个 CPU buffer
    for phase in self.cpu_buffers[0]:
        phase.load_state_dict_from_disk(0, None)
    for phase in self.cpu_buffers[1]:
        phase.load_state_dict_from_disk(1, None)
```

预热的作用是让操作系统把权重文件的磁盘块缓存到 page cache 中，后续读取更快。

### 5.8 Disk Offload 的完整调用链

在 Phase 级 Offload 的基础上，增加磁盘→CPU 的预取层：

```
for block_idx in range(40):
    │
    ├── ① start_prefetch_block(block_idx+1)     ← 后台线程
    │     └── ThreadPoolExecutor.submit():
    │           磁盘读取 block_{N+1}.safetensors → cpu_buffers[1]
    │           (多个 phase 文件可并行读取)
    │
    ├── ② infer_phases(block_idx):               ← 正常 Phase 级 Offload
    │     for phase_idx in range(3):
    │         ├── [最后一个 phase] swap_cpu_buffers()
    │         │     └── 等 ① 完成，交换 cpu_buffers[0] 和 [1]
    │         ├── prefetch_phase()                ← CPU → GPU (cuda_load_stream)
    │         ├── infer_phase()                   ← GPU 计算 (compute_stream)
    │         └── swap_phases()
    │
    └── (下一个 block 的磁盘预取已在 ① 的后台线程中进行)

三级并行：
─────────────────────────────────────────────────────────
磁盘→CPU:   [读 Block N+1]        [读 Block N+2]
CPU→GPU:         [传 Phase]  [传 Phase]  [传 Phase]
GPU 计算:        [SelfAttn] [CrossAttn] [FFN]
─────────────────────────────────────────────────────────
```

**GPU 显存峰值**：同 Phase 级（~4.2GB）
**CPU 内存**：只需要 2 个 Block 的 CPU buffer（FP8 ~770MB），而非整个模型

---

## 6. model 级 Offload

model 级是最简单的 Offload 粒度，只用于 **Wan 2.2 MoE 双模型切换**（第3课详细讲过）：

```python
# wan/model.py:160-166
def infer(self, inputs):
    if self.cpu_offload:
        if self.offload_granularity == "model":
            # 第一个 step 时把整个模型搬到 GPU
            if self.scheduler.step_index == 0:
                self.to_cuda()
        elif self.offload_granularity != "model":
            # block/phase 粒度：只搬小权重，Block 由 offload_manager 管理
            self.pre_weight.to_cuda()
            self.transformer_weights.non_block_weights_to_cuda()
```

在 MoE 场景中，`MultiModelStruct.get_current_model_index()` 根据 timestep 决定使用哪个模型，并在切换时做整体的 CPU↔GPU 搬运（第3课第5节）。

---

## 7. 实际配置示例

### 7.1 Block Offload（4090 24GB）

```json
{
    "cpu_offload": true,
    "offload_granularity": "block",
    "dit_quantized": true,
    "dit_quant_scheme": "fp8-q8f",
    "t5_cpu_offload": false,
    "clip_cpu_offload": false,
    "vae_cpu_offload": false
}
```

DiT 用 block Offload + FP8 量化，编码器和 VAE 不 offload（24GB 够放编码器+2个FP8 Block）。

### 7.2 Phase Offload（16GB 显卡）

```json
{
    "cpu_offload": true,
    "offload_granularity": "phase",
    "dit_quantized": true,
    "dit_quant_scheme": "fp8-q8f",
    "t5_cpu_offload": false,
    "clip_cpu_offload": false,
    "vae_cpu_offload": false
}
```

比 block 更省显存，代价是传输次数 ×3。

### 7.3 Disk Offload（CPU 内存也不够）

```json
{
    "cpu_offload": true,
    "offload_granularity": "phase",
    "lazy_load": true,
    "dit_quantized": true,
    "dit_quant_scheme": "fp8-vllm",
    "t5_cpu_offload": true,
    "clip_cpu_offload": true,
    "vae_cpu_offload": true,
    "use_tae": true
}
```

全部 offload + 量化 + TAE 轻量 VAE，适合极端低资源环境。

---

## 8. 为新模型添加 Offload 支持

如果你要适配的新模型已经按照第4课的 infer/weights 分离原则实现，添加 Offload 只需要：

### 步骤1：权重层注册缓冲区

```python
# my_model/weights/transformer_weights.py
class MyTransformerWeights(WeightModule):
    def __init__(self, config, ...):
        self.blocks = WeightModuleList([MyBlock(i, ...) for i in range(num_layers)])
        self.register_offload_buffers(config, ...)  # ← 复用现有逻辑

    def register_offload_buffers(self, config, ...):
        # 直接参考 WanTransformerWeights.register_offload_buffers
        if config["cpu_offload"]:
            if config["offload_granularity"] == "block":
                self.offload_block_cuda_buffers = WeightModuleList([
                    MyBlock(i, create_cuda_buffer=True, ...) for i in range(2)
                ])
```

### 步骤2：推理层继承 Offload 推理类

```python
# my_model/infer/offload/transformer_infer.py
from lightx2v.common.offload.manager import WeightAsyncStreamManager

class MyOffloadTransformerInfer(MyTransformerInfer):
    def __init__(self, config):
        super().__init__(config)
        if config.get("cpu_offload", False):
            granularity = config.get("offload_granularity", "block")
            if granularity == "block":
                self.infer_func = self.infer_with_blocks_offload
            elif granularity == "phase":
                self.infer_func = self.infer_with_phases_offload
            self.offload_manager = WeightAsyncStreamManager(granularity)
```

如果新模型的 Block 内部结构和 Wan 类似（SelfAttn + CrossAttn + FFN），可以直接继承 `WanOffloadTransformerInfer` 复用 `infer_with_blocks_offload` 和 `infer_with_phases_offload`。

### 步骤3：Model 层选择推理类

```python
# my_model/model.py
def _init_infer_class(self):
    if not self.cpu_offload:
        self.transformer_infer_class = MyTransformerInfer
    else:
        self.transformer_infer_class = MyOffloadTransformerInfer
```

---

## 9. 关键源码文件索引

| 文件 | 行号 | 内容 |
|------|------|------|
| **WeightAsyncStreamManager** | | |
| `common/offload/manager.py` | 14-26 | 初始化，CUDA Stream 创建 |
| `common/offload/manager.py` | 28-48 | init_cpu_buffer / init_cuda_buffer |
| `common/offload/manager.py` | 62-75 | init_first_buffer |
| `common/offload/manager.py` | 77-82 | prefetch_weights（Block 级异步传输） |
| `common/offload/manager.py` | 84-89 | prefetch_phase（Phase 级异步传输） |
| `common/offload/manager.py` | 91-107 | swap_blocks / swap_phases（同步+交换） |
| `common/offload/manager.py` | 109-122 | warm_up_cpu_buffers（磁盘预热） |
| `common/offload/manager.py` | 124-149 | lazy_load 初始化 + 磁盘预取 + CPU buffer 交换 |
| **WanOffloadTransformerInfer** | | |
| `wan/infer/offload/transformer_infer.py` | 10-37 | 初始化，粒度选择 |
| `wan/infer/offload/transformer_infer.py` | 39-71 | infer_with_blocks_offload |
| `wan/infer/offload/transformer_infer.py` | 73-114 | infer_with_phases_offload + infer_phases |
| `wan/infer/offload/transformer_infer.py` | 116-155 | infer_phase（phase 内部计算） |
| **权重缓冲区创建** | | |
| `wan/weights/transformer_weights.py` | 55-132 | register_offload_buffers |
| `wan/weights/transformer_weights.py` | 134-142 | non_block_weights_to_cuda/cpu |
| **Model 层 Offload 逻辑** | | |
| `wan/model.py` | 159-199 | infer() 中的 Offload 入口/出口 |
| **配置示例** | | |
| `configs/offload/block/wan_i2v_block.json` | | Block Offload 配置 |
| `configs/offload/phase/wan_i2v_phase.json` | | Phase Offload 配置 |
| `configs/offload/disk/wan_i2v_phase_lazy_load_480p.json` | | Disk Offload 配置 |

---

## 10. 思考题

1. **为什么 block Offload 用双缓冲（2 个 buffer），而 phase Offload 用 3 个 buffer（每个 phase 一个）？它们的交换逻辑有什么区别？**
   - 提示：对比 `swap_blocks()` 和 `swap_phases()` 的实现

2. **lazy_load 的磁盘预取为什么用 `ThreadPoolExecutor` 而不是 CUDA Stream？**
   - 提示：想想磁盘 IO 和 CPU↔GPU DMA 传输的区别

3. **phase 级 Offload 中 `phase_params` 为什么用字典而不是局部变量？**
   - 提示：考虑三个 phase 的计算是分开调用的，不在同一个函数栈帧中

4. **在什么情况下 phase 级 Offload 反而比 block 级更慢？**
   - 提示：回顾第4课的讨论——计算时间 vs 传输时间

---

## 下一课预告

第7课将深入注意力算子体系，分析：
- 注意力算子的统一接口设计（AttnWeightTemplate）
- Flash Attention 2/3 的集成方式
- Sage Attention 的集成方式
- 算子选择机制：attn_mode 参数的路由
- 如何集成新的注意力算子
