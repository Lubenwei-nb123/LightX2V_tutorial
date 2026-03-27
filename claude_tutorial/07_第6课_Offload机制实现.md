# 第6课：Offload 机制实现

## 学习目标
- 深入理解 Offload 机制涉及的底层数据结构
- 理解为什么框架要自定义 WeightModule 而非使用 nn.Module
- 掌握三种 Offload 粒度的推理调用链
- 通过实际权重文件分析，理解为什么不同模型支持不同的 Offload 粒度

---

## 第一部分：底层数据结构（自底向上）

## 1. 为什么不用 nn.Module

PyTorch 的 `nn.Module` 是为**训练**设计的，它带有大量推理不需要的功能：
- 自动求导图的构建与管理
- 参数注册到优化器（`named_parameters()`、`parameters()`）
- Hook 机制（forward_hook、backward_hook）
- 梯度检查点（gradient checkpointing）

同时 `nn.Module` **缺少**推理框架需要的能力：

| 需求 | nn.Module | WeightModule |
|------|-----------|--------------|
| 按 block_index 参数化加载权重 | 不支持 | `load_state_dict(dict, block_index)` |
| 支持 FP8/INT8 等非标准张量类型 | `nn.Parameter` 只接受标准 Tensor | 直接存储任意张量 |
| 权重 CPU↔GPU 搬运（含 pin_memory） | `module.to(device)` 会遍历所有参数 | 精确控制每个张量的搬运 |
| 从磁盘按需加载单个 Block | 不支持 | `load_state_dict_from_disk(block_index)` |
| 同构 buffer 复用（不同 Block 写入同一 buffer） | 不支持 | `resolve_block_name` 机制 |

**核心矛盾**：Offload 的双缓冲需要把**不同 Block 的权重**加载到**同一个 GPU buffer** 中。nn.Module 的 `load_state_dict` 按 key 严格匹配，无法实现"buffer 0 这一次装 Block 5、下一次装 Block 6"的动态映射。WeightModule 通过 `resolve_block_name(name, block_index)` 解决了这个问题。

---

## 2. WeightModule：权重容器基类

**核心文件**：`lightx2v/common/modules/weight_module.py`

```python
class WeightModule:
    """所有权重容器的基类——一个轻量的树形容器"""

    def __init__(self):
        self._modules = {}      # 子模块（嵌套的 WeightModule）
        self._parameters = {}   # 直接持有的权重（叶子节点）

    # ==================== 注册 ====================
    def add_module(self, name, module):
        """注册子模块，同时设为属性（self.xxx 可访问）"""
        self._modules[name] = module
        setattr(self, name, module)

    def register_parameter(self, name, param):
        """注册参数（叶子权重节点）"""
        self._parameters[name] = param
        setattr(self, name, param)

    # ==================== 权重加载 ====================
    def load(self, weight_dict):
        """从权重字典递归加载所有子模块和参数"""
        for _, module in self._modules.items():
            if hasattr(module, "load"):
                module.load(weight_dict)     # 向下递归
        for _, parameter in self._parameters.items():
            if hasattr(parameter, "load"):
                parameter.load(weight_dict)  # 叶子节点加载

    # ==================== Offload 核心方法 ====================
    def state_dict(self, destination=None):
        """导出所有权重为 {name: tensor} 字典"""
        if destination is None:
            destination = {}
        for _, param in self._parameters.items():
            if param is not None:
                param.state_dict(destination)    # 叶子导出
        for _, module in self._modules.items():
            if module is not None:
                module.state_dict(destination)   # 递归导出
        return destination

    def load_state_dict(self, destination, block_index, adapter_block_index=None):
        """从字典加载权重——Offload buffer 间拷贝的核心方法
        block_index 参数让同一个 buffer 可以加载不同 Block 的权重"""
        for _, param in self._parameters.items():
            if param is not None:
                param.load_state_dict(destination, block_index, adapter_block_index)
        for _, module in self._modules.items():
            if module is not None:
                module.load_state_dict(destination, block_index, adapter_block_index)

    def load_state_dict_from_disk(self, block_index, adapter_block_index=None):
        """从磁盘加载权重（lazy_load 用）"""
        for _, param in self._parameters.items():
            if param is not None:
                param.load_state_dict_from_disk(block_index, adapter_block_index)
        for _, module in self._modules.items():
            if module is not None:
                module.load_state_dict_from_disk(block_index, adapter_block_index)

    # ==================== 设备迁移 ====================
    def to_cpu(self, non_blocking=False):
        """递归地把所有权重搬到 CPU"""
        for name, param in self._parameters.items():
            if param is not None:
                if hasattr(param, "cpu"):
                    self._parameters[name] = param.cpu()
                    setattr(self, name, self._parameters[name])
                elif hasattr(param, "to_cpu"):
                    self._parameters[name].to_cpu()
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
        """递归地把所有权重搬到 GPU（结构同 to_cpu）"""
        # ... 方向相反，用 param.to(AI_DEVICE) ...
```

### WeightModuleList：有序列表容器

```python
class WeightModuleList(WeightModule):
    """可索引的 WeightModule 列表，类似 nn.ModuleList"""
    def __init__(self, modules=None):
        super().__init__()
        self._list = []
        if modules is not None:
            for idx, module in enumerate(modules):
                self.append(module)

    def append(self, module):
        idx = len(self._list)
        self._list.append(module)
        self.add_module(str(idx), module)   # 用字符串索引注册为子模块

    def __getitem__(self, idx):   return self._list[idx]
    def __setitem__(self, idx, module):
        self._list[idx] = module
        self.add_module(str(idx), module)
    def __len__(self):           return len(self._list)
    def __iter__(self):          return iter(self._list)
```

---

## 3. 叶子节点：MMWeight 的三种形态

叶子节点是真正持有张量的对象。以 `MMWeight`（矩阵乘法权重，第4课注册器机制）为例，它通过 `create_cuda_buffer` / `create_cpu_buffer` 标志呈现三种形态：

```python
# common/ops/mm/mm_weight.py（简化）
@MM_WEIGHT_REGISTER("Default")
class MMWeight(MMWeightTemplate):
    def __init__(self, weight_name, bias_name=None,
                 create_cuda_buffer=False, create_cpu_buffer=False,
                 lazy_load=False, lazy_load_file=None, ...):
        self.weight_name = weight_name   # 如 "blocks.0.self_attn.q.weight"
        self.bias_name = bias_name

    def load(self, weight_dict):
        if not self.create_cuda_buffer and not self.create_cpu_buffer and not self.lazy_load:
            # ===== 形态1：普通权重 =====
            # 从 weight_dict 加载真实权重
            self.weight = weight_dict[self.weight_name].t()    # 转置存储
            self.bias = weight_dict.get(self.bias_name, None)
            # 同时保存 pin_memory 版本（Offload 传输时用，DMA 更快）
            self.pin_weight = self.weight.pin_memory() if self.weight.is_cpu else None

        elif self.create_cuda_buffer:
            # ===== 形态2：GPU buffer（空壳）=====
            # 只分配 GPU 上的空张量，等着被 copy_ 填入数据
            self.weight_cuda_buffer = torch.empty(shape, device="cuda")
            self.bias_cuda_buffer = ...

        elif self.create_cpu_buffer:
            # ===== 形态3：CPU buffer（pin_memory 空壳）=====
            # 分配页锁定 CPU 内存（lazy_load 磁盘预取的中转站）
            self.pin_weight = torch.empty(shape).pin_memory()
            self.weight = None

    def apply(self, input_tensor):
        """推理时调用——不管哪种形态，接口一致"""
        return torch.addmm(self.bias, input_tensor, self.weight)
```

**`state_dict()` 和 `load_state_dict()` 的叶子实现**：

```python
# common/ops/utils.py:286-339（叶子节点最终调用的函数）

def state_dict(cls, base_attrs, lora_attrs, destination=None):
    """导出权重到字典"""
    if destination is None:
        destination = {}
    for _, base_attr, _ in base_attrs:
        # 优先导出 pin_memory 版本（传输更快）
        pin_attr = getattr(cls, f"pin_{base_attr}", None)
        device_attr = getattr(cls, base_attr, None)
        name = getattr(cls, f"{base_attr}_name")
        destination[name] = pin_attr if pin_attr is not None else device_attr
    return destination

def load_state_dict(cls, base_attrs, lora_attrs, destination, block_index, ...):
    """从字典加载权重到 GPU buffer"""
    for name, attr_name, _ in base_attrs:
        # 关键：resolve_block_name 把 buffer 的通用名解析为具体 block 的名
        # "blocks.0.self_attn.q.weight" + block_index=5
        #  → "blocks.5.self_attn.q.weight"
        actual_name = resolve_block_name(name, block_index, ...)
        cuda_buffer_attr = f"{attr_name}_cuda_buffer"
        if actual_name in destination:
            # 核心：GPU buffer.copy_(CPU tensor, non_blocking=True)
            setattr(cls, attr_name,
                getattr(cls, cuda_buffer_attr).copy_(
                    destination[actual_name], non_blocking=True
                ))
```

**三种形态的总结**：

| 标志 | 形态 | 持有的张量 | 用途 |
|------|------|-----------|------|
| 全 False | 普通权重 | `weight`（设备上）+ `pin_weight`（CPU pin_memory） | 存储实际权重 |
| `create_cuda_buffer=True` | GPU buffer | `weight_cuda_buffer`（GPU 空张量） | 接收 Offload 传输 |
| `create_cpu_buffer=True` | CPU buffer | `pin_weight`（CPU pin_memory 空张量） | lazy_load 磁盘中转 |

---

## 4. WanTransformerAttentionBlock：Block 的内部结构

`WanTransformerAttentionBlock` 是一个 `WeightModule`，内部通过 `compute_phases` 组织了 3 个 Phase：

```python
# wan/weights/transformer_weights.py:145-215（简化）
class WanTransformerAttentionBlock(WeightModule):
    def __init__(self, block_index, task, mm_type, config,
                 create_cuda_buffer=False, create_cpu_buffer=False,
                 lazy_load=False, lazy_load_path=None, lora_path=None):
        super().__init__()
        # mm_type 通过注册器决定矩阵乘法实现（Default/fp8-vllm/...）
        self.compute_phases = WeightModuleList([
            WanSelfAttention(block_index, "blocks", task, mm_type, config,
                             create_cuda_buffer, create_cpu_buffer, ...),
            WanCrossAttention(block_index, "blocks", task, mm_type, config,
                              create_cuda_buffer, create_cpu_buffer, ...),
            WanFFN(block_index, "blocks", task, mm_type, config,
                    create_cuda_buffer, create_cpu_buffer, ...),
        ])
        self.add_module("compute_phases", self.compute_phases)
```

**WanSelfAttention 内部**（展示叶子节点的注册方式）：

```python
# wan/weights/transformer_weights.py:218-390（简化）
class WanSelfAttention(WeightModule):
    def __init__(self, block_index, block_prefix, task, mm_type, config,
                 create_cuda_buffer, create_cpu_buffer, lazy_load, ...):
        super().__init__()
        # 调制参数（小张量）
        self.add_module("modulation", TENSOR_REGISTER["Default"](
            f"{block_prefix}.{block_index}.modulation",
            create_cuda_buffer, create_cpu_buffer, lazy_load, ...))
        # LayerNorm
        self.add_module("norm1", LN_WEIGHT_REGISTER["torch"]())
        # Q/K/V/O 投影（大矩阵，用 mm_type 选择量化实现）
        self.add_module("self_attn_q", MM_WEIGHT_REGISTER[mm_type](
            f"{block_prefix}.{block_index}.self_attn.q.weight",
            f"{block_prefix}.{block_index}.self_attn.q.bias",
            create_cuda_buffer, create_cpu_buffer, lazy_load, ...))
        self.add_module("self_attn_k", MM_WEIGHT_REGISTER[mm_type](...))
        self.add_module("self_attn_v", MM_WEIGHT_REGISTER[mm_type](...))
        self.add_module("self_attn_o", MM_WEIGHT_REGISTER[mm_type](...))
        # QK Norm
        self.add_module("self_attn_norm_q", RMS_WEIGHT_REGISTER[rms_type](...))
        self.add_module("self_attn_norm_k", RMS_WEIGHT_REGISTER[rms_type](...))
        # 注意力算子（无权重，通过注册器选择 flash_attn2/sage_attn 等）
        self.add_module("self_attn_1", ATTN_WEIGHT_REGISTER[config["self_attn_1_type"]]())
```

**注意**：`create_cuda_buffer` 和 `create_cpu_buffer` 标志会**逐层向下传递**。当 WanTransformerAttentionBlock 以 `create_cuda_buffer=True` 创建时，内部所有 MMWeight、TensorWeight 等叶子节点也都以 `create_cuda_buffer=True` 创建——它们分配的都是 GPU 上的空张量，而非真实权重。

---

## 5. WanTransformerWeights：完整的权重体系

```python
# wan/weights/transformer_weights.py:11-142（简化）
class WanTransformerWeights(WeightModule):
    def __init__(self, config, lazy_load_path=None, lora_path=None):
        super().__init__()
        self.blocks_num = config["num_layers"]   # 40
        self.mm_type = config.get("dit_quant_scheme", "Default")
        self.lazy_load = config.get("lazy_load", False)

        # ① 40 个 Block 的实际权重
        self.blocks = WeightModuleList([
            WanTransformerAttentionBlock(
                block_index=i, mm_type=self.mm_type, config=config,
                create_cuda_buffer=False, create_cpu_buffer=False,
                lazy_load=self.lazy_load, lazy_load_path=lazy_load_path,
            )
            for i in range(self.blocks_num)
        ])

        # ② Offload 缓冲区（根据粒度创建）
        self.register_offload_buffers(config, lazy_load_path, lora_path)
        self.add_module("blocks", self.blocks)

        # ③ 非 Block 权重（不参与逐 Block Offload）
        self.register_parameter("norm", LN_WEIGHT_REGISTER["torch"]())
        self.add_module("head", MM_WEIGHT_REGISTER["Default"](
            "head.head.weight", "head.head.bias"))
        self.register_parameter("head_modulation",
            TENSOR_REGISTER["Default"]("head.modulation"))

    def non_block_weights_to_cuda(self):
        """每个 step 开始时搬小权重到 GPU"""
        self.norm.to_cuda()
        self.head.to_cuda()
        self.head_modulation.to_cuda()

    def non_block_weights_to_cpu(self):
        """每个 step 结束后搬回 CPU"""
        self.norm.to_cpu()
        self.head.to_cpu()
        self.head_modulation.to_cpu()
```

### register_offload_buffers：按粒度创建缓冲区

```python
    def register_offload_buffers(self, config, lazy_load_path, lora_path):
        if config["cpu_offload"]:
            if config["offload_granularity"] == "block":
                # ---- Block 级：2 个完整 Block 的 GPU buffer ----
                self.offload_block_cuda_buffers = WeightModuleList([
                    WanTransformerAttentionBlock(
                        block_index=i, mm_type=self.mm_type, config=config,
                        create_cuda_buffer=True,   # ← GPU 空壳
                    )
                    for i in range(2)              # 双缓冲
                ])
                self.offload_phase_cuda_buffers = None

                if self.lazy_load:
                    # lazy_load 还需要 CPU buffer（磁盘→CPU 的中转站）
                    self.offload_block_cpu_buffers = WeightModuleList([
                        WanTransformerAttentionBlock(
                            block_index=i, create_cpu_buffer=True, ...
                        )
                        for i in range(2)          # CPU 也双缓冲
                    ])

            elif config["offload_granularity"] == "phase":
                # ---- Phase 级：只取 compute_phases 作为 GPU buffer ----
                self.offload_phase_cuda_buffers = WanTransformerAttentionBlock(
                    block_index=0, create_cuda_buffer=True, ...
                ).compute_phases                   # ← 取 3 个 Phase
                # 创建了完整 Block 空壳，但只保留 compute_phases（SelfAttn/CrossAttn/FFN）
                self.offload_block_cuda_buffers = None

                if self.lazy_load:
                    self.offload_phase_cpu_buffers = WeightModuleList([
                        WanTransformerAttentionBlock(
                            block_index=i, create_cpu_buffer=True, ...
                        ).compute_phases
                        for i in range(2)          # CPU 双缓冲
                    ])
```

### 完整数据结构一览

以 **Block 级 Offload + lazy_load** 为例：

```
WanTransformerWeights
│
├── blocks (WeightModuleList, len=40)              ← 实际权重
│     └── [i] WanTransformerAttentionBlock
│           └── compute_phases (WeightModuleList, len=3)
│                 ├── [0] WanSelfAttention
│                 │     ├── self_attn_q: MMWeight  → weight=None (lazy_load)
│                 │     │                            weight_name="blocks.0.self_attn.q.weight"
│                 │     ├── self_attn_k, self_attn_v, self_attn_o: MMWeight
│                 │     ├── norm1: LNWeight
│                 │     ├── modulation: TensorWeight
│                 │     └── self_attn_1: FlashAttn2Weight（无权重）
│                 ├── [1] WanCrossAttention（结构类似）
│                 └── [2] WanFFN（结构类似）
│
├── offload_block_cuda_buffers (WeightModuleList, len=2)  ← GPU 双缓冲
│     └── [j] WanTransformerAttentionBlock(create_cuda_buffer=True)
│           └── compute_phases
│                 └── [0] WanSelfAttention
│                       └── self_attn_q: MMWeight
│                             weight_cuda_buffer = torch.empty(5120, 5120, device="cuda")
│                             ↑ GPU 上的空张量，等着 copy_ 填入数据
│
├── offload_block_cpu_buffers (WeightModuleList, len=2)   ← CPU 双缓冲
│     └── [k] WanTransformerAttentionBlock(create_cpu_buffer=True)
│           └── self_attn_q: MMWeight
│                 pin_weight = torch.empty(5120, 5120).pin_memory()
│                 ↑ 页锁定 CPU 内存，DMA 传输更快
│
├── norm, head, head_modulation                           ← 非 Block 小权重
```

---

## 6. WeightAsyncStreamManager：异步传输管理器

**核心文件**：`lightx2v/common/offload/manager.py`

这是 Offload 的运行时核心——管理 CUDA Stream、异步传输和双缓冲交换。

```python
class WeightAsyncStreamManager(object):
    def __init__(self, offload_granularity):
        self.offload_granularity = offload_granularity
        # 三个独立的 CUDA Stream
        self.init_stream = torch_device_module.Stream(priority=0)      # 初始化用
        self.cuda_load_stream = torch_device_module.Stream(priority=1) # CPU→GPU 传输
        self.compute_stream = torch_device_module.Stream(priority=-1)  # GPU 计算
        self.need_init_first_buffer = True

    def init_cuda_buffer(self, blocks_cuda_buffer=None, phases_cuda_buffer=None):
        """设置 GPU 缓冲区引用"""
        if self.offload_granularity == "block":
            self.cuda_buffers = [blocks_cuda_buffer[i] for i in range(len(blocks_cuda_buffer))]
        elif self.offload_granularity == "phase":
            self.cuda_buffers = [phases_cuda_buffer[i] for i in range(len(phases_cuda_buffer))]

    def prefetch_weights(self, block_idx, blocks):
        """异步传输：在 cuda_load_stream 上把下一个 Block 从 CPU 拷到 GPU buffer"""
        with torch_device_module.stream(self.cuda_load_stream):
            # state_dict() 导出源权重 → load_state_dict() 拷贝到 buffer
            # 内部最终调用 cuda_buffer.copy_(cpu_tensor, non_blocking=True)
            self.cuda_buffers[1].load_state_dict(
                self.cpu_buffers[0].state_dict() if hasattr(self, "cpu_buffers")
                else blocks[block_idx].state_dict(),
                block_idx)

    def prefetch_phase(self, block_idx, phase_idx, blocks):
        """异步传输：Phase 级，只传一个 Phase"""
        with torch_device_module.stream(self.cuda_load_stream):
            self.cuda_buffers[phase_idx].load_state_dict(
                blocks[block_idx].compute_phases[phase_idx].state_dict()
                if not hasattr(self, "cpu_buffers")
                else self.cpu_buffers[0][phase_idx].state_dict(),
                block_idx)

    def swap_blocks(self):
        """同步两个 Stream，然后交换双缓冲"""
        self.cuda_load_stream.synchronize()  # 等传输完成
        self.compute_stream.synchronize()     # 等计算完成
        self.cuda_buffers[0], self.cuda_buffers[1] = (
            self.cuda_buffers[1], self.cuda_buffers[0])

    def swap_phases(self):
        """同步 Stream（phase 不需要交换 buffer，因为有 3 个独立 buffer）"""
        self.cuda_load_stream.synchronize()
        self.compute_stream.synchronize()

    # ===== lazy_load 专用 =====
    def init_lazy_load(self, num_workers=6):
        self.lazy_load = True
        self.executor = ThreadPoolExecutor(max_workers=num_workers)

    def start_prefetch_block(self, block_idx):
        """后台线程从磁盘读取下一个 Block 到 CPU buffer"""
        self.prefetch_futures = []
        if self.offload_granularity == "block":
            future = self.executor.submit(
                self.cpu_buffers[1].load_state_dict_from_disk, block_idx)
            self.prefetch_futures.append(future)
        else:
            for phase in self.cpu_buffers[1]:
                future = self.executor.submit(
                    phase.load_state_dict_from_disk, block_idx)
                self.prefetch_futures.append(future)

    def swap_cpu_buffers(self):
        """等磁盘预取完成，交换 CPU 双缓冲"""
        for f in self.prefetch_futures:
            f.result()  # 阻塞等待
        self.cpu_buffers = [self.cpu_buffers[1], self.cpu_buffers[0]]
```

---

## 第二部分：各 Offload 模式的推理调用链

## 7. 显存压力分析

Wan 2.1 14B (I2V, dim=5120, 40层) 的权重分布：

```
组件                      BF16 运行时    FP8 运行时    占比
─────────────────────────────────────────────────────────
DiT Block 权重（40 层）    ~30 GB        ~15 GB       86%
  - 每个 Block             ~770 MB       ~385 MB
DiT 非 Block 权重          ~450 MB       ~450 MB      1%
T5 编码器                  ~5 GB                      7%
CLIP 编码器                ~1.5 GB                    2%
VAE                        ~0.5 GB                    1%
激活值 + latent            ~4 GB                      -
─────────────────────────────────────────────────────────
```

## 8. 无 Offload 调用链

```
Pipeline.generate()
 └── Runner.run_pipeline(input_info)
      ├── run_input_encoder()
      │     ├── T5.infer()   ← 常驻 GPU
      │     ├── CLIP.visual() ← 常驻 GPU
      │     └── VAE.encode() ← 常驻 GPU
      └── run_main()
           └── for step:
                 ├── scheduler.step_pre()
                 ├── model.infer(inputs)
                 │     └── transformer_infer.infer_without_offload():
                 │           for block_idx in range(40):
                 │               infer_block(blocks[block_idx], ...)
                 │               └── 所有 Block 权重常驻 GPU
                 └── scheduler.step_post()
```

**GPU 峰值**：BF16 ~35GB / FP8 ~20GB

## 9. Block 级 Offload 调用链

```
model.infer(inputs)
 ├── pre_weight.to_cuda()                    ← 小权重搬到 GPU
 ├── transformer_weights.non_block_weights_to_cuda()
 │
 ├── _infer_cond_uncond()
 │     └── transformer_infer.infer_with_blocks_offload():
 │           │
 │           ├── init_first_buffer()          ← 首次：Block 0 → GPU buffer[0]
 │           │
 │           └── for block_idx in range(40):
 │                 ├── [lazy_load] start_prefetch_block(N+1)
 │                 │     └── ThreadPoolExecutor: 磁盘 → cpu_buffers[1]
 │                 ├── [lazy_load] swap_cpu_buffers()
 │                 │
 │                 ├── prefetch_weights(N+1)
 │                 │     └── cuda_load_stream: cpu → cuda_buffers[1]
 │                 │
 │                 ├── compute_stream:
 │                 │     infer_block(cuda_buffers[0], x, ...)
 │                 │     └── SelfAttn → CrossAttn → FFN（连续执行）
 │                 │
 │                 └── swap_blocks()
 │                       ├── cuda_load_stream.synchronize()
 │                       ├── compute_stream.synchronize()
 │                       └── buffer[0] ↔ buffer[1]
 │
 ├── pre_weight.to_cpu()
 └── transformer_weights.non_block_weights_to_cpu()
```

**GPU 峰值**：FP8 ~5GB（2 个 FP8 Block ~770MB + 小权重 + 激活 + latent）

## 10. Phase 级 Offload 调用链

```
transformer_infer.infer_with_phases_offload():
 └── for block_idx in range(40):
       └── infer_phases(block_idx):
             ├── phase_idx=0 (SelfAttention):
             │     ├── prefetch_phase(block, phase=1)
             │     │     └── cuda_load_stream: CrossAttn 权重 → buffer[1]
             │     ├── compute_stream:
             │     │     infer_phase(0, buffer[0], ...)
             │     │     └── 结果存入 phase_params["y_out"]
             │     └── swap_phases()
             │
             ├── phase_idx=1 (CrossAttention):
             │     ├── prefetch_phase(block, phase=2)
             │     │     └── cuda_load_stream: FFN 权重 → buffer[2]
             │     ├── compute_stream:
             │     │     infer_phase(1, buffer[1], ...)
             │     │     └── 读 phase_params["y_out"]，写 phase_params["attn_out"]
             │     └── swap_phases()
             │
             └── phase_idx=2 (FFN):
                   ├── prefetch_phase(block+1, phase=0)
                   │     └── 下一个 Block 的 SelfAttn → buffer[0]
                   ├── compute_stream:
                   │     infer_phase(2, buffer[2], ...)
                   │     └── 读 phase_params["attn_out"]
                   └── swap_phases()
```

**GPU 峰值**：FP8 ~4.2GB（最大 FP8 Phase ~150MB + 激活 + latent）

**`phase_params` 字典**：因为三个 phase 在不同的函数调用中执行，局部变量无法传递中间结果，需要一个持久化的字典在 phase 之间共享状态。

---

## 第三部分：模型差异分析——为什么 Wan 有 Phase Offload 而 LTX 没有

## 11. Wan vs LTX2 的 Block 内部结构对比

### 11.1 数据来源

**Wan 2.1 I2V 14B**：
- 权重文件：`/data/phd/hf_models/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-*.safetensors`
- 配置文件：`/data/phd/hf_models/Wan2.1-I2V-14B-480P/config.json`

```json
// config.json 内容
{
  "dim": 5120,
  "ffn_dim": 13824,
  "num_heads": 40,
  "num_layers": 40,
  "in_dim": 36,
  "out_dim": 16,
  "text_len": 512
}
```

**LTX 2.3 22B**：
- 权重文件：`/data/wengzicheng/LTX-2.3/ltx-2.3-22b-distilled.safetensors`

### 11.2 实测数据

以下数据通过遍历 safetensors 文件中 Block 0 的所有 key，按子模块名分组统计 `tensor.numel() * tensor.element_size()` 得到。

**Wan 2.1 Block 0**（权重存储为 FP32，3 个 Phase）：

```
Phase                        存储大小    参数量     FP8 运行时
──────────────────────────────────────────────────────────
Phase 0 (SelfAttn+norm1+mod)  400 MB    104.9M    ~100 MB
Phase 1 (CrossAttn+norm3)     600 MB    157.3M    ~150 MB
Phase 2 (FFN+norm2)           540 MB    141.6M    ~135 MB
──────────────────────────────────────────────────────────
Block 合计                    1540 MB   403.8M    ~385 MB
最大/最小 Phase 比值：1.50
```

**LTX 2.3 Block 0**（权重存储为 BF16，8 个 Phase）：

```
子模块                         存储大小    占 Block 比例
──────────────────────────────────────────────────────
ff (视频 FFN)                  256.0 MB    34.7%
attn1 (视频 SelfAttn)          128.3 MB    17.4%
attn2 (视频 CrossAttn)         128.3 MB    17.4%
audio_ff (音频 FFN)             64.0 MB     8.7%
audio_to_video_attn             48.3 MB     6.5%
video_to_audio_attn             48.1 MB     6.5%
audio_attn1 (音频 SelfAttn)     32.1 MB     4.4%
audio_attn2 (音频 CrossAttn)    32.1 MB     4.4%
scale_shift_table 等（6个小张量） 0.4 MB     <0.1%
──────────────────────────────────────────────────────
Block 合计                     737.8 MB    100%
最大/最小子模块比值：16386（忽略小张量后仍有 8 倍）
```

### 11.3 为什么 LTX 不适合 Phase Offload

Phase Offload 的前提是**各 Phase 大小接近**，这样：
1. GPU buffer 大小 ≈ 最大 Phase → 越均匀，buffer 越小
2. 每个 Phase 的计算时间接近 → 传输更容易被计算掩盖

| 维度 | Wan 2.1 | LTX 2.3 |
|------|---------|---------|
| Phase 数量 | 3 | 8 |
| 最大 Phase | 150 MB (FP8) | 256 MB (BF16) |
| 最小 Phase | 100 MB (FP8) | 32 MB (BF16) |
| 最大/最小比值 | **1.5** | **8.0** |
| 传输次数（每 Block） | 3 | 8 |

**LTX 的问题**：

1. **Phase 大小极度不均**：`ff`（256MB）是 `audio_attn1`（32MB）的 8 倍。GPU buffer 必须按最大 Phase 分配，小 Phase 浪费 buffer 空间。

2. **传输次数过多**：8 个 Phase 意味着每个 Block 要做 8 次 CPU→GPU 传输。小 Phase 的计算量很小（32MB 权重的矩阵乘法几乎瞬间完成），传输根本来不及被计算掩盖。

3. **计算-传输重叠效果差**：第3课讨论过，计算时间越短越难掩盖传输。LTX 的小 Phase（32MB）计算时间可能只有 1-2ms，但传输一个 256MB 的大 Phase 需要 10+ms。

4. **实现复杂度高**：Wan 的 3 个 Phase 结构统一（SelfAttn/CrossAttn/FFN），phase_params 只需要传递 6 个调制参数 + y_out + attn_out。LTX 的 8 个 Phase 包含视频/音频两套路径 + 跨模态注意力，中间状态的传递逻辑远比 Wan 复杂。

**看代码的证据**：

```python
# ltx2/weights/transformer_weights.py:45-66
def register_offload_buffers(self, config, lazy_load_path, lora_path):
    if config["cpu_offload"]:
        if config["offload_granularity"] == "block":
            # 只实现了 block 级，没有 phase 级
            self.offload_block_cuda_buffers = WeightModuleList([
                LTX2TransformerBlock(create_cuda_buffer=True, ...)
                for i in range(2)
            ])
            self.offload_phase_cuda_buffers = None
            # 注意：没有 elif "phase" 分支

# ltx2/infer/offload/transformer_infer.py:42-50
if offload_granularity == "block":
    self.infer_func = self.infer_with_blocks_offload
elif offload_granularity == "model":
    self.infer_func = self.infer_without_offload
else:
    raise ValueError(f"Unsupported offload_granularity: {offload_granularity}")
    # 明确不支持 phase
```

### 11.4 LTX2 能否通过合并子模块实现 Phase Offload？

虽然 LTX2 的 8 个子模块大小悬殊，但如果**按计算顺序合并**为 3 组，比值会大幅改善：

```
当前 LTX2 Block（8 个子模块，按计算顺序排列）：
  attn1 (128MB) → attn2 (128MB) → audio_attn1 (32MB) → audio_attn2 (32MB)
  → audio_to_video (48MB) → video_to_audio (48MB) → ff (256MB) → audio_ff (64MB)

合并方案（3 个 Phase）：
  Phase 0: attn1 + attn2                                   = 256 MB  (视频注意力)
  Phase 1: audio_attn1 + audio_attn2 + a2v_attn + v2a_attn = 160 MB  (音频+跨模态)
  Phase 2: ff + audio_ff                                    = 320 MB  (全部 FFN)

比值：256 : 160 : 320 ≈ 1.6 : 1 : 2  → 远优于合并前的 8:1
```

这个方案**技术上完全可行**，但当前框架没有实现，原因是工程层面的：

1. **Phase Offload 的 `infer_phase()` 与 Wan 的 3-phase 结构耦合**。Wan 的 `infer_phase()` 硬编码了 `phase_idx=0` 是 SelfAttn、`phase_idx=1` 是 CrossAttn、`phase_idx=2` 是 FFN（见 `wan/infer/offload/transformer_infer.py:116-155`）。LTX2 的合并方案需要一套独立的 `infer_phase` 逻辑。

2. **`compute_phases` 的粒度是固定的**。LTX2 的 `LTX2TransformerBlock.compute_phases` 直接包含 8 个独立 WeightModule。要合并为 3 个 Phase，需要在中间引入一个分组层，把多个子模块包成一组 WeightModule，同时修改 `state_dict()` / `load_state_dict()` 的遍历逻辑。

3. **LTX2 是较新加入框架的模型**，block Offload 已经能满足大部分需求，phase Offload 还没有排上优先级。

**如果要实现，关键改动是**：让 `compute_phases` 支持"分组"概念——一个 Phase 可以包含多个子模块，Offload 时按分组传输，推理时按分组内的顺序逐个执行。这是一个值得做的框架层面的泛化。

### 11.5 结论

Phase Offload 是一个**依赖模型架构特性**的优化，不是通用的。它需要：

1. Block 内部可以自然地拆分为少数几个大小接近的子模块
2. 子模块之间的中间状态简单，容易跨 Phase 传递
3. 每个子模块的计算量足够大，能掩盖下一个子模块的传输

Wan 的 SelfAttn/CrossAttn/FFN 三分结构天然满足这三个条件（第4.2 节分析过，比值只有 1.5），所以框架为它实现了 Phase Offload。LTX 的 8-way 多模态结构不满足条件 1 和 3，所以只提供 Block Offload。

---

## 12. 实际配置示例

### Block Offload（4090 24GB）

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

### Phase Offload（16GB 显卡）

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

### Disk Offload（CPU 内存也紧张）

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

---

## 13. 思考题

1. **WeightModule 的 `state_dict()` 优先导出 `pin_weight` 而非 `weight`，为什么？**
   - 提示：考虑 CPU→GPU DMA 传输对内存类型的要求

2. **Block 级 Offload 的 `swap_blocks()` 为什么要同时等待 `cuda_load_stream` 和 `compute_stream`？只等一个行不行？**
   - 提示：考虑两个 stream 操作的是同一对 buffer

3. **Phase 级 Offload 为什么用 `swap_phases()`（只同步不交换）而 Block 级用 `swap_blocks()`（同步+交换）？**
   - 提示：对比两者的 buffer 数量和使用方式

4. **如果要给 LTX2 添加 Phase Offload，你会怎么设计 Phase 的划分？**
   - 提示：考虑把大小接近的子模块合并

---

## 下一课预告

第7课将深入注意力算子体系，分析：
- 注意力算子的统一接口设计（AttnWeightTemplate）
- Flash Attention 2/3 的集成方式
- Sage Attention 的集成方式
- 算子选择机制：attn_mode 参数的路由
