# 附录A：Python 包管理机制与 LightX2V 工程实践

## 学习目标

完成本课后，你应该能够：

- 区分模块、导入包、发行包、项目和环境
- 理解 `import`、`sys.path`、`sys.modules` 与 `__init__.py` 的协作方式
- 理解 `pyproject.toml`、PEP 517、PEP 518、PEP 621、wheel 和 sdist
- 区分普通安装、可编辑安装、源码运行和 Git 安装
- 理解依赖声明、依赖解析、版本约束、锁文件和可选依赖
- 理解 Python 包如何携带 C++、CUDA、SYCL 等本地扩展
- 理解 entry point 如何实现不修改主仓库的插件扩展
- 能够读懂 LightX2V 的包边界、导入副作用、注册器与平台插件机制
- 能够判断一个大型框架的包管理设计哪些地方成熟，哪些地方仍需改进

---

## 0. 为什么学习框架前必须理解包管理

很多看似是“模型代码”的问题，本质上其实是包管理问题：

- 仓库中明明有这个文件，为什么 `import` 失败？
- 为什么在仓库根目录能运行，换一个目录就失败？
- 为什么 `pip install -e .` 后修改源码立即生效？
- 为什么安装了 `lightx2v`，却没有 `lightx2v_kernel`？
- 为什么配置里写了 `flash_attn3`，导入时只打印缺少依赖，真正运行时才报错？
- 为什么只执行 `import lightx2v` 就开始检测 GPU 平台？
- 为什么某个 Runner 文件存在，却没有出现在 `RUNNER_REGISTER` 中？
- 为什么发行包名是 `lightx2v-kernel`，导入名却是 `lightx2v_kernel`？
- 为什么当前环境可以运行，但换一台机器重新安装后得到不同版本组合？

理解这些问题，需要先把“包”这个词拆开。

---

# 第一部分：Python 包管理的基本对象

## 1. 五个容易混淆的概念

### 1.1 模块 module

一个可被 Python 加载的代码单元通常称为模块。最常见形式是单个 `.py` 文件：

```text
lightx2v/pipeline.py
```

对应导入名：

```python
import lightx2v.pipeline
```

模块也可能由本地扩展提供，例如 `.so`、`.pyd`，不一定是 Python 源文件。

### 1.2 导入包 import package

包是可以包含子模块的导入单元，通常表现为目录：

```text
lightx2v/
├── __init__.py
├── pipeline.py
├── models/
└── utils/
```

对应导入名：

```python
import lightx2v
from lightx2v import pipeline
```

`lightx2v` 是导入包，`lightx2v.pipeline` 是包中的模块。

### 1.3 发行包 distribution package

发行包是安装工具处理的制品，它拥有名称、版本、依赖和元数据。

LightX2V 根目录 `pyproject.toml` 声明：

```toml
[project]
name = "lightx2v"
version = "0.1.0"
```

这里的 `lightx2v` 是发行包名。它被构建后通常得到类似：

```text
lightx2v-0.1.0-py3-none-any.whl
```

发行包名和导入名不要求相同。例如：

```text
发行包名：lightx2v-kernel
导入包名：lightx2v_kernel
```

对应源码：

- `LightX2V/lightx2v_kernel/pyproject.toml:8-12`
- `LightX2V/lightx2v_kernel/pyproject.toml:33-40`

安装和导入是两个不同动作：

```bash
python -m pip install lightx2v-kernel
```

```python
import lightx2v_kernel
```

### 1.4 项目 project

项目是开发者维护的整个源码仓库。一个项目可以构建一个或多个发行包。

LightX2V 仓库至少包含三套独立构建边界：

```text
LightX2V/
├── pyproject.toml                 → lightx2v
├── lightx2v_kernel/pyproject.toml → lightx2v-kernel
└── lightx2v_kernel_xpu/pyproject.toml → sycl-kernels
```

因此：

> 一个 Git 仓库不一定只对应一个 Python 发行包。

### 1.5 环境 environment

环境是某个 Python 解释器及其可见依赖的集合，包括：

- Python 版本
- `site-packages`
- 安装的发行包及版本
- 环境变量
- 动态链接库
- CUDA、ROCm、XPU 等运行时

查看当前解释器：

```bash
python -c "import sys; print(sys.executable)"
```

查看发行包：

```bash
python -m pip show lightx2v
python -m pip list
```

推荐使用 `python -m pip`，因为它明确表示“使用这个 Python 对应的 pip”，可避免 `python` 和 `pip` 指向不同环境。

---

## 2. 一张总图

```text
Git 项目
  │
  ├─ pyproject.toml
  ├─ Python 源码
  ├─ 配置和资源
  └─ C++/CUDA/SYCL 源码
          │
          ▼
PEP 517 构建后端
          │
     ┌────┴────┐
     ▼         ▼
   sdist      wheel
源码发行包    二进制/可安装发行包
     │         │
     └────┬────┘
          ▼
pip/uv 安装到某个 Python 环境
          │
          ▼
site-packages + distribution metadata
          │
          ▼
Python import system 按导入名加载模块
```

构建系统负责“如何把项目做成制品”，安装器负责“如何放进环境”，导入系统负责“运行时如何找到模块”。三者有关联，但不是同一层。

---

# 第二部分：Python 导入机制

## 3. 执行 `import` 时发生了什么

以：

```python
from lightx2v import LightX2VPipeline
```

为例，可以用以下流程理解：

```text
1. 检查 sys.modules 中是否已有 lightx2v
2. 若没有，按 import finder 顺序查找 lightx2v
3. finder 在 sys.path 中寻找包或扩展模块
4. 创建 module 对象并先放入 sys.modules
5. 执行 lightx2v/__init__.py
6. 从包对象中读取 LightX2VPipeline
7. 将名称绑定到当前模块命名空间
```

先放入 `sys.modules` 再执行模块代码，是 Python 支持递归导入的关键，但也解释了循环导入时为什么可能看到“部分初始化的模块”。

### 3.1 `sys.path` 从哪里来

常见来源包括：

- 启动脚本所在目录
- 当前工作目录，取决于启动方式
- `PYTHONPATH`
- 标准库路径
- 当前环境的 `site-packages`
- `.pth` 文件加入的路径
- 可编辑安装生成的导入映射

检查实际搜索路径：

```bash
python -c "import sys; print('\n'.join(sys.path))"
```

检查模块最终从哪里加载：

```bash
python -c "import lightx2v; print(lightx2v.__file__)"
```

这两个命令是排查“导入了错误副本”的第一工具。

### 3.2 `sys.modules` 是进程级模块缓存

```python
import sys
import lightx2v

print(sys.modules["lightx2v"])
```

同一解释器进程内再次导入，通常不会重新执行模块顶层代码。它会复用 `sys.modules` 中的对象。

这对注册器非常重要：

- 第一次导入模块时执行装饰器，完成注册
- 第二次普通导入不会再次执行注册
- 删除源码文件不会自动清除进程中的已加载模块
- 在 Notebook 中改代码后，仅重新运行 `import` 不一定加载新代码

因此调试注册机制时，最可靠的方法通常是启动新 Python 进程，而不是依赖复杂的 `reload()`。

### 3.3 `__init__.py` 的三种职责

`__init__.py` 常承担：

1. 标记普通包
2. 组织公共 API
3. 执行包初始化逻辑

LightX2V 顶层文件：

- `LightX2V/lightx2v/__init__.py:1-18`

它做了三件重要的事：

```python
import lightx2v_platform.set_ai_device
from lightx2v import common, models, utils
from lightx2v.pipeline import LightX2VPipeline
```

这意味着 `import lightx2v` 不是纯粹的名称声明，它会：

- 初始化硬件平台
- 导入框架子包
- 暴露 `LightX2VPipeline`

`__all__` 主要控制 `from lightx2v import *` 暴露哪些名称，也能表达公共 API 意图，但它不构成访问控制。未列入 `__all__` 的对象仍可能通过完整路径导入。

### 3.4 导入副作用

模块顶层执行的动作称为导入副作用，例如：

- 读取环境变量
- 初始化硬件
- 注册实现类
- 打印日志
- 加载动态库
- 建立网络连接

LightX2V 中：

```python
import lightx2v_platform.set_ai_device
```

最终会执行：

```python
set_ai_device()
```

位置：

- `LightX2V/lightx2v_platform/set_ai_device.py:41-53`

平台名从环境变量读取：

```text
PLATFORM=cuda
PLATFORM=xpu
PLATFORM=npu
```

如果只是做静态检查而没有对应设备，可以看到代码还提供了 `SKIP_PLATFORM_CHECK`，但它只跳过可用性失败，不改变平台注册逻辑：

- `LightX2V/lightx2v_platform/base/base.py:9-36`

成熟框架需要控制导入副作用，因为过重的顶层导入会带来：

- `import` 变慢
- 可选依赖更容易在启动时失败
- 文档工具和类型检查工具更难运行
- 单元测试更难隔离
- 环境变量必须在导入前设置

LightX2V 用顶层初始化换取了全局设备状态的一致性，这是明确的工程权衡，不是无成本设计。

---

## 4. 绝对导入、相对导入和循环导入

### 4.1 绝对导入

```python
from lightx2v.utils.registry_factory import RUNNER_REGISTER
```

优点：

- 来源清晰
- 跨包重构时容易搜索
- 不依赖当前模块的相对层级

### 4.2 相对导入

```python
from .main import run_server
```

LightX2V Server 入口使用了这种方式：

- `LightX2V/lightx2v/server/__main__.py:1-3`

相对导入适合包内紧密关联的模块，但文件不能被当成普通脚本随意直接执行。推荐：

```bash
python -m lightx2v.server
```

而不是：

```bash
python lightx2v/server/__main__.py
```

`python -m` 会先按包语义解析模块，正确设置 `__package__`，因此相对导入可以工作。

### 4.3 循环导入为什么出现

假设：

```text
a.py 导入 b.py
b.py 又在顶层导入 a.py 中尚未定义的名称
```

此时 `a` 已进入 `sys.modules`，但尚未执行完毕，`b` 看到的是“部分初始化的 a”。

常见治理方式：

- 将共享类型下沉到第三个模块
- 只导入模块，在函数执行时访问属性
- 对仅用于类型标注的导入使用 `TYPE_CHECKING`
- 将可选依赖延迟到实际使用处导入
- 减少 `__init__.py` 中的全量 re-export
- 不用 `sys.path` 修改掩盖结构问题

---

## 5. 普通包与命名空间包

### 5.1 普通包

带 `__init__.py` 的目录通常是普通包。其边界明确，初始化行为可控。

### 5.2 隐式命名空间包

Python 3 支持没有 `__init__.py` 的 namespace package。多个发行包可以共同提供同一个顶层命名空间。

这适合大型插件生态，例如：

```text
company_plugins.audio
company_plugins.video
```

但自动包发现也可能意外把不希望发布的目录纳入 wheel。

LightX2V 使用：

```toml
[tool.setuptools.packages.find]
include = ["lightx2v*"]
exclude = ["lightx2v_kernel*"]
```

位置：

- `LightX2V/pyproject.toml:80-86`

这个匹配不只覆盖 `lightx2v`，也会匹配：

```text
lightx2v_platform
lightx2v_ros
lightx2v_train
```

实际用 setuptools 的 namespace package finder 验证，当前会发现这些顶层目录及其大量子目录。

这体现了两面性：

- 优点：单一 wheel 可以携带推理、平台层及其他配套模块
- 风险：源码目录只要以 `lightx2v` 开头，就可能被意外打入发行包

更严格的成熟项目通常会：

- 使用 `src/` layout 隔离项目文件和可导入包
- 精确列出要发布的包
- 为训练、ROS、kernel 建立独立发行包
- 在 CI 中检查 wheel 文件清单

---

# 第三部分：从源码到可安装制品

## 6. `pyproject.toml` 的三层职责

### 6.1 `[build-system]`：谁来构建

LightX2V 根项目：

```toml
[build-system]
requires = [
    "setuptools>=61.0",
    "wheel",
    "packaging",
    "ninja",
]
build-backend = "setuptools.build_meta"
```

位置：

- `LightX2V/pyproject.toml:1-8`

这对应 PEP 517 和 PEP 518：

- 前端工具：`pip`、`build`、`uv`
- 构建后端：`setuptools.build_meta`
- 隔离构建环境所需依赖：`build-system.requires`

构建前端不需要理解 setuptools 内部实现，只按标准钩子调用后端。

### 6.2 `[project]`：发行包元数据

PEP 621 将名称、版本、Python 要求、依赖等统一放进标准表：

```toml
[project]
name = "lightx2v"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "numpy",
    "torch<=2.8.0",
    "transformers",
]
```

位置：

- `LightX2V/pyproject.toml:10-78`

`pip install .` 时，安装器会读取这些元数据并解析运行时依赖。

### 6.3 `[tool.*]`：工具专属配置

例如：

```toml
[tool.setuptools.packages.find]
include = ["lightx2v*"]
exclude = ["lightx2v_kernel*"]

[tool.ruff]
target-version = "py311"
```

标准只规定命名空间，具体字段由对应工具解释。

---

## 7. PEP 517 隔离构建

执行：

```bash
python -m pip wheel .
```

概念上会发生：

```text
1. 读取 pyproject.toml
2. 创建临时隔离构建环境
3. 安装 build-system.requires
4. 加载 build-backend
5. 请求后端生成 wheel metadata
6. 解析并安装项目运行依赖
7. 构建 wheel
```

构建依赖和运行依赖要分开：

| 类型 | 用途 | LightX2V 示例 |
|---|---|---|
| 构建依赖 | 生成 wheel | setuptools、wheel、ninja |
| 运行依赖 | 用户运行包 | torch、transformers、fastapi |
| 开发依赖 | 测试、lint、文档 | ruff、pre-commit、pytest |
| 可选后端依赖 | 特定优化路径 | flash-attn、SageAttention、vllm |

把运行依赖错误地写入构建依赖，会导致每次构建都安装不必要的大包；把构建时必须 import 的依赖只写到运行依赖，则隔离构建可能失败。

---

## 8. wheel 与 sdist

### 8.1 wheel

wheel 是已构建的安装制品，扩展名为 `.whl`。安装时通常不再执行完整编译。

纯 Python wheel 常见文件名：

```text
lightx2v-0.1.0-py3-none-any.whl
```

本地扩展 wheel 可能是：

```text
sycl_kernels-0.0.1-cp311-cp311-win_amd64.whl
```

标签表达：

- Python ABI
- 平台
- 架构

### 8.2 sdist

sdist 是源码发行包，通常是 `.tar.gz`。安装时需要在目标环境构建 wheel，因此更依赖：

- 编译器
- CMake/Ninja
- CUDA 或 SYCL 工具链
- 系统头文件
- 构建隔离环境

对于大型 GPU 框架，优先提供经过测试的 wheel 可以显著降低安装失败率。

### 8.3 如何检查 wheel 内容

推荐构建后检查：

```bash
python -m build
python -m zipfile -l dist/lightx2v-0.1.0-py3-none-any.whl
```

或安装 `check-wheel-contents` 做自动检查。

重点确认：

- 是否误带训练数据、测试文件和大模型权重
- 配置文件是否被包含
- `lightx2v_platform` 是否在 wheel 中
- 不应打包的 ROS 或训练目录是否被纳入
- license 和 metadata 是否完整

---

## 9. 普通安装、可编辑安装与源码目录运行

### 9.1 普通安装

```bash
python -m pip install .
```

构建 wheel 后将其安装到环境。之后修改源码目录，已安装副本通常不会自动变化。

适合：

- 部署
- CI 验证
- 接近用户安装方式的测试

### 9.2 可编辑安装

```bash
python -m pip install -e .
```

PEP 660 定义了现代可编辑安装接口。它通常通过导入映射或 `.pth` 文件，让环境中的导入指向工作区源码。

适合：

- 框架开发
- 调试新模型
- 修改后立即验证

但要注意：

- 修改 Python 文件通常立即生效，但当前进程仍受 `sys.modules` 缓存影响
- 修改包元数据、依赖、entry point 后通常需要重新安装
- 修改 C++/CUDA 扩展后通常需要重新编译
- 可编辑安装成功不代表普通 wheel 一定正确

LightX2V 文档推荐开发环境使用：

```bash
python -m pip install -v -e .
```

位置：

- `LightX2V/docs/ZH_CN/source/getting_started/quickstart.md:82-90`

### 9.3 Git URL 安装

```bash
python -m pip install -v git+https://github.com/ModelTC/LightX2V.git
```

位置：

- `LightX2V/README_zh.md:157-160`

生产环境应固定 commit 或 tag，避免 main 更新后环境不可复现：

```bash
python -m pip install "git+https://github.com/ModelTC/LightX2V.git@<commit>"
```

### 9.4 直接在仓库根目录运行

在仓库根目录执行脚本时，当前目录可能位于 `sys.path`，即使没有安装也能导入源码。

这容易制造假象：

> 在仓库里能运行，不等于 wheel 安装后能运行。

成熟项目应同时测试：

1. 源码开发模式
2. 可编辑安装
3. 构建 wheel 后在干净环境安装
4. 从仓库外目录导入和执行

---

# 第四部分：依赖管理

## 10. 依赖声明与依赖解析不是一回事

项目只声明约束：

```text
torch<=2.8.0
transformers
numpy
```

安装器会把所有直接依赖和传递依赖的约束合并，求出一个可安装版本集合。

如果只写：

```text
transformers
```

不同日期安装可能得到不同版本。即使项目源码没有变化，行为也可能变化。

### 10.1 版本约束的常见形式

```text
package==1.2.3
package>=1.2
package<2
package>=1.2,<2
package~=1.4
package!=1.5.0
```

含义：

| 形式 | 典型用途 |
|---|---|
| `==` | 已验证的严格版本 |
| `>=` | 要求最低能力 |
| `<` | 排除不兼容的大版本或版本段 |
| `>=,<` | 推荐的兼容窗口 |
| `~=` | 兼容版本发布范围 |
| `!=` | 排除已知坏版本 |

核心原则：

- 库通常声明兼容范围，不应无理由把所有传递依赖钉死
- 应用和部署环境需要锁定完整依赖图
- GPU 栈需同时约束 Python 包、驱动、运行时与硬件架构

### 10.2 直接依赖与传递依赖

如果 LightX2V 直接 import `torch`，`torch` 就应是直接依赖。不能因为某个其他库碰巧依赖 torch，就依赖它“顺带安装”。

否则对方移除依赖后，LightX2V 环境会突然缺包。

### 10.3 可选依赖

大型推理框架通常不应强迫所有用户安装全部后端。可以使用：

```toml
[project.optional-dependencies]
server = ["fastapi", "uvicorn", "redis"]
docs = ["sphinx", "myst-parser"]
cuda = ["flash-attn"]
train = ["swanlab", "peft"]
```

用户按需安装：

```bash
python -m pip install ".[server]"
```

当前 LightX2V 根项目把推理、Web UI、Server、数据库、对象存储和训练相关库大量放在基础 `dependencies` 中：

- `LightX2V/pyproject.toml:33-75`

这使“一条命令安装完整功能”较简单，但代价是：

- 基础安装更重
- 冲突面更大
- 部署镜像更大
- 某个边缘依赖失败会阻塞普通推理安装

因此这部分是当前实现的便利性取舍，而不是包管理的最终形态。更优雅的方向是按功能拆分 extras。

### 10.4 环境标记

平台差异可以在依赖中表达：

```toml
dependencies = [
    "some-package; sys_platform == 'linux'",
]
```

但 CUDA wheel 来源、驱动版本和厂商运行时往往超出普通 Python metadata 能完整表达的范围，仍需要：

- 专用 index
- constraints 文件
- Docker 镜像
- 平台安装文档
- CI 构建矩阵

---

## 11. `pyproject.toml` 与 `requirements.txt`

LightX2V 同时存在：

- `LightX2V/pyproject.toml`
- `LightX2V/requirements.txt`
- `requirements_win.txt`
- `requirements_animate.txt`
- `requirements-docs.txt`

两份主依赖声明目前并不完全一致。例如 `requirements.txt` 包含某些 `pyproject.toml` 未包含的包，反之亦然。

应先明确每个文件的角色：

| 文件 | 推荐角色 |
|---|---|
| `pyproject.toml` | 发行包直接运行依赖与 extras |
| `requirements*.txt` | 特定部署场景或完整环境输入 |
| `constraints*.txt` | 限制解析版本但不主动引入包 |
| lock 文件 | 某个平台和 Python 版本的完整解析结果 |

如果多个文件都被当作“权威依赖清单”，它们会逐渐漂移。

更稳妥的方案是建立单一事实源，然后自动生成场景化文件。

---

## 12. 锁文件与可复现环境

### 12.1 为什么版本范围还不够

假设项目声明：

```text
transformers>=4.40
```

今天和三个月后解析出的版本可能不同，传递依赖也可能不同。

锁文件记录一组已解析的精确版本，有些工具还记录 wheel hash 和平台信息。

常见工具：

- `uv lock`
- Poetry lock
- PDM lock
- pip-tools 生成 pinned requirements
- Conda lock

### 12.2 库与应用的不同策略

LightX2V 既像库，又像可直接部署的应用，适合采用双层策略：

```text
发行 metadata：声明兼容范围
部署 lock/constraints：固定验证过的完整环境
```

建议按平台维护环境矩阵：

```text
Python 3.11 + CUDA 12.x + torch 2.8
Python 3.11 + Intel XPU + 对应 torch
Python 3.11 + Ascend NPU + 对应 torch_npu
Windows + CUDA + 对应 wheel 集
```

单个跨平台锁文件未必能覆盖所有 GPU 生态，可能需要每个平台独立锁定。

### 12.3 当前 LightX2V 的状态

当前仓库未看到根级 `uv.lock`、`poetry.lock` 或 `pdm.lock`。这意味着：

- `pyproject.toml` 提供可安装约束
- 文档和 Docker 提供场景化环境经验
- 但仅凭源码仓库无法得到完全锁定的 Python 依赖图

因此生产部署应记录：

```bash
python -m pip freeze
```

更理想的是由 CI 生成并验证平台 constraints，而不是部署后临时 freeze。

---

## 13. 可选依赖的运行时处理

LightX2V 的注意力实现采用了常见的延迟失败模式：

```python
try:
    from flash_attn import flash_attn_func
except ImportError:
    flash_attn_func = None
```

位置：

- `LightX2V/lightx2v/common/ops/attn/flash_attn.py:1-29`

这种做法允许未安装 Flash Attention 的用户仍导入框架，并选择其他算子。

但必须区分四个状态：

```text
实现模块存在
    ≠ 第三方依赖已安装
    ≠ 算子已成功注册且可实例化
    ≠ 当前硬件和输入形状可执行
```

成熟的可选依赖处理应做到：

- 导入主包不因未选择的后端失败
- 实际选择后端时给出清晰错误
- 错误信息说明安装哪个 extra 或 wheel
- 能力检测和配置验证尽量提前
- CI 覆盖“安装”和“未安装”两种路径

---

# 第五部分：本地扩展与多发行包设计

## 14. 为什么 kernel 应独立成发行包

纯 Python 框架与本地 kernel 的生命周期不同：

| 维度 | Python 主包 | 本地 kernel 包 |
|---|---|---|
| 构建工具 | setuptools | CMake + scikit-build-core |
| 平台相关性 | 相对较低 | 强相关 |
| ABI | 通常无 CPython ABI | 受 Python/torch/CUDA 影响 |
| 发布频率 | 功能迭代 | 内核与硬件优化迭代 |
| 安装失败率 | 较低 | 编译环境敏感 |
| wheel 数量 | 少 | 按平台和 ABI 多份 |

LightX2V 将 CUDA kernel 独立为：

```text
lightx2v_kernel/
├── pyproject.toml
├── CMakeLists.txt
└── python/lightx2v_kernel/
```

其构建后端：

```toml
[build-system]
requires = [
  "scikit-build-core>=0.10",
  "torch>=2.7.0",
  "wheel",
]
build-backend = "scikit_build_core.build"
```

位置：

- `LightX2V/lightx2v_kernel/pyproject.toml:1-6`

XPU kernel 又是独立发行包：

- 发行名：`sycl-kernels`
- 导入名：`sycl_kernels`
- 编译器：`icx`
- C++ 标准：C++20

位置：

- `LightX2V/lightx2v_kernel_xpu/pyproject.toml:1-29`

这种拆分的优点：

- 主包可独立安装
- 不同硬件只安装自己的 kernel
- kernel wheel 可以独立发布
- 构建失败不必阻塞纯 Python 开发
- 避免主 wheel 同时绑定所有厂商工具链

### 14.1 wheel 的 ABI 与平台标签

本地扩展 wheel 必须考虑：

- CPython 版本，例如 `cp311`
- Python ABI，例如 `abi3` 或 CPython 专属 ABI
- 操作系统与架构
- torch ABI
- CUDA、ROCm、SYCL 运行时
- GPU 架构

`wheel.py-api = "cp39"` 不等价于“所有 Python 版本都一定兼容”，还需要扩展本身遵守对应 ABI 规则，并经过实际测试。

### 14.2 构建时依赖 torch 的代价

kernel 构建环境需要 torch 头文件和 CMake 配置时，会把 torch 放入 `build-system.requires`。

优点：隔离构建能获得必需头文件。

代价：

- 构建环境可能重新下载巨大的 torch wheel
- 若 build isolation 中 torch 版本与运行环境不同，可能产生 ABI 风险
- GPU 专用 index 不一定被隔离环境正确使用

因此本地扩展经常需要：

- 预构建 wheel
- 明确 torch 兼容矩阵
- 必要时使用 `--no-build-isolation`
- 在容器中构建
- CI 对最终 wheel 做真实 import 和 kernel smoke test

---

# 第六部分：插件、注册器与包边界

## 15. Python entry point 是什么

entry point 是发行包 metadata 中声明的“字符串名称 → Python 对象”映射。

一个第三方平台包可以声明：

```toml
[project.entry-points."lightx2v.platform_plugins"]
my_accelerator = "my_lightx2v_plugin:register"
```

安装后，主框架无需 import 固定包名，就可以通过 metadata 发现它：

```python
from importlib.metadata import entry_points

for ep in entry_points(group="lightx2v.platform_plugins"):
    ep.load()()
```

entry point 的关键价值是：

> 扩展关系由已安装发行包的 metadata 建立，而不是要求用户修改主仓库源码。

### 15.1 LightX2V 的平台插件加载

当前主分支已经实现 out-of-tree 平台插件：

- `LightX2V/lightx2v_platform/set_ai_device.py:8-39`

加载顺序是：

```text
import lightx2v
  → import lightx2v_platform.set_ai_device
  → 发现 lightx2v.platform_plugins entry points
  → 加载并执行每个插件的注册函数
  → init_ai_device(platform)
  → import lightx2v_platform.ops
  → 框架注册器 merge 平台注册器
```

这是一个成熟框架很重要的扩展能力：

- 第三方硬件厂商可以独立发包
- 主仓库不必内置所有后端
- 插件可以注册 Device、Attention、MM、Norm、RoPE 实现
- 插件失败会记录 warning 并跳过，而不是直接阻断所有平台初始化

### 15.2 为什么加载顺序非常重要

平台侧注册器定义于：

- `LightX2V/lightx2v_platform/registry_factory.py:1-73`

框架侧注册器在导入时执行一次 merge：

- `LightX2V/lightx2v/utils/registry_factory.py:69-88`

因此插件必须在框架侧执行 merge 之前完成注册。否则平台实现已经进入平台表，却没有进入框架使用的表。

这说明：

> 包管理中的导入顺序可以直接影响运行时能力集合。

### 15.3 插件包的最小结构

```text
my_lightx2v_plugin/
├── pyproject.toml
└── src/
    └── my_lightx2v_plugin/
        ├── __init__.py
        └── plugin.py
```

`pyproject.toml` 概念示例：

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "my-lightx2v-plugin"
version = "0.1.0"
dependencies = ["lightx2v>=0.1.0"]

[project.entry-points."lightx2v.platform_plugins"]
my_device = "my_lightx2v_plugin.plugin:register"
```

注册函数应：

- 无参数
- 幂等或明确处理重复注册
- 不在 import 阶段分配大显存
- 不吞掉关键错误
- 注册平台类和平台算子
- 对主框架版本声明兼容范围

---

## 16. entry point 和框架内部装饰器注册的区别

LightX2V 同时使用两种注册机制。

### 16.1 发行包级发现：entry point

用于发现“环境中安装了哪些外部插件”。

```text
范围：跨发行包
数据来源：dist-info metadata
触发方式：importlib.metadata.entry_points
```

### 16.2 进程内路由：装饰器注册器

用于把已经导入的 Python 类放入运行时表：

```python
@RUNNER_REGISTER("wan2.1")
class WanRunner:
    pass
```

```text
范围：当前 Python 进程
数据来源：模块顶层代码执行
触发方式：import 对应模块
```

二者可以组合：

```text
entry point 找到外部发行包
  → 加载插件函数
  → 插件函数 import 自己的实现
  → 装饰器或 register() 写入运行时注册表
```

### 16.3 “文件存在”为什么不等于“已经注册”

LightX2V CLI 在 `infer.py` 顶部显式导入大量 Runner：

- `LightX2V/lightx2v/infer.py:8-49`

这些导入有时看起来“没有使用”，但实际作用是执行类定义上的装饰器。

因此：

```text
runner.py 文件存在
  ≠ 模块被导入
  ≠ 装饰器已执行
  ≠ RUNNER_REGISTER 中有对应 key
```

Ruff 配置对 `__init__.py` 的 `F401` 做了忽略，也与“导入用于 API 暴露或注册副作用”这种模式有关：

- `LightX2V/pyproject.toml:104-108`

---

# 第七部分：LightX2V 的包管理全景

## 17. 当前仓库的主要包边界

```text
LightX2V/
├── pyproject.toml
├── lightx2v/                 主推理框架
│   ├── pipeline.py
│   ├── infer.py
│   ├── models/
│   ├── common/
│   ├── server/
│   └── utils/
├── lightx2v_platform/        硬件平台抽象与平台算子
├── lightx2v_train/           训练相关源码和配置
├── lightx2v_ros/             ROS 集成
├── lightx2v_kernel/          CUDA kernel 独立构建项目
└── lightx2v_kernel_xpu/      SYCL/XPU kernel 独立构建项目
```

注意：目录边界、导入包边界和发行包边界并不完全相同。

根 `setuptools` 自动发现规则会把多个 `lightx2v*` 目录视为候选包；两个 kernel 目录被显式排除，并由各自 `pyproject.toml` 构建。

### 17.1 主包公共 API

用户推荐入口：

```python
from lightx2v import LightX2VPipeline
```

顶层 `__all__` 暴露：

- 版本元数据
- `models`
- `common`
- `utils`
- `LightX2VPipeline`

位置：

- `LightX2V/lightx2v/__init__.py:1-18`

这是 facade 模式：用户不必知道 `pipeline.py` 的内部位置。

### 17.2 模块执行入口

Server 支持：

```bash
python -m lightx2v.server --model_path ... --model_cls ...
```

原因是包内提供了：

```text
lightx2v/server/__main__.py
```

位置：

- `LightX2V/lightx2v/server/__main__.py:1-32`

但根 `pyproject.toml` 当前没有 `[project.scripts]`，因此安装后并未通过标准 metadata 声明类似 `lightx2v-server` 的 console script。

可以改进为：

```toml
[project.scripts]
lightx2v-infer = "lightx2v.infer:main"
lightx2v-server = "lightx2v.server.__main__:main"
```

前提是目标模块提供稳定、无参数的 `main()` 入口，并明确 CLI 兼容策略。

### 17.3 平台包为什么独立于 `lightx2v` 命名空间

`lightx2v_platform` 有自己的注册器和设备抽象。这样做可以：

- 在框架公共注册器建立前先完成平台初始化
- 将厂商差异与模型代码隔离
- 为外部平台插件提供稳定注册目标
- 避免模型层直接判断每一种硬件

代价是顶层包数量增加，发行包发现规则必须更精确。

### 17.4 kernel 为什么不放进主 wheel

根配置显式排除：

```toml
exclude = ["lightx2v_kernel*"]
```

位置：

- `LightX2V/pyproject.toml:83-86`

否则 setuptools 可能把 kernel 的 Python 源目录当普通包处理，却不会自动完成正确的 CMake 编译和平台 wheel 标记。

独立构建边界保证 kernel 使用自己的：

- 构建后端
- 版本
- Python ABI
- 平台标签
- 编译器配置

---

## 18. LightX2V 做得好的地方

### 18.1 使用标准化的现代构建入口

主包使用 `pyproject.toml`、PEP 517 后端和 PEP 621 metadata，不依赖用户直接执行仓库中的 `setup.py`。

### 18.2 主包与本地 kernel 分离

CUDA 和 XPU kernel 使用独立的 `scikit-build-core` 项目，避免把所有平台编译链耦合到主包。

### 18.3 平台层与模型层隔离

`lightx2v_platform` 提供统一设备抽象，框架注册器再 merge 平台算子，模型不必硬编码所有厂商。

### 18.4 支持 out-of-tree 插件

`lightx2v.platform_plugins` entry point 允许外部发行包扩展硬件后端，减少主仓库修改。

### 18.5 可选高性能算子采用延迟失败

没有安装 Flash Attention 时，用户仍可导入框架并选择其他 backend，而不是主包导入立即失败。

### 18.6 公共 API 相对集中

顶层暴露 `LightX2VPipeline`，用户入口简洁，内部模块仍可按职责组织。

### 18.7 import 副作用服务于注册机制

Runner 和算子通过导入触发装饰器注册，避免核心路由维护巨大 `if/elif`。

---

## 19. 仍可改进的地方

成熟并不意味着没有技术债。以当前主分支为准，以下方向值得改进。

### 19.1 基础依赖过宽

基础 `dependencies` 同时包含：

- 核心推理
- Web UI
- Server
- 消息队列
- 数据库
- 对象存储
- 训练和监控相关库

建议拆分：

```text
lightx2v
lightx2v[server]
lightx2v[ui]
lightx2v[train]
lightx2v[storage]
lightx2v[all]
```

### 19.2 缺少根级锁文件或已发布 constraints

版本范围适合库 metadata，但部署需要平台化锁定结果。

建议：

- 为支持矩阵生成 constraints
- CI 用相同 constraints 重建环境
- Docker 镜像记录完整包版本
- 对 wheel 使用 hash 校验

### 19.3 `requirements.txt` 与 `pyproject.toml` 有漂移风险

应指定唯一事实源并自动生成其他文件，避免人工维护多份重叠列表。

### 19.4 版本号存在重复事实源

当前：

```text
pyproject.toml: version = 0.1.0
lightx2v/__init__.py: __version__ = 0.1.0
```

人工同步容易出错。可选择：

- 从 `importlib.metadata.version("lightx2v")` 获取
- 使用 setuptools dynamic version
- 使用基于 Git tag 的版本工具

### 19.5 自动包发现范围偏大

`include = ["lightx2v*"]` 会覆盖多个顶层目录。建议：

- 精确声明包
- 使用 `src/` layout
- 将训练、ROS 拆成独立发行包
- CI 检查 wheel 内容

### 19.6 缺少标准 console scripts

当前可使用 `python -m`，但没有 `[project.scripts]`。增加稳定命令入口可以改善安装后的用户体验。

### 19.7 顶层 import 较重

`import lightx2v` 会初始化平台并导入多个子包。可评估：

- 延迟导入
- 将硬件初始化移到显式 API
- 保留轻量 metadata import
- 为文档、静态分析和 CPU 环境提供更稳定入口

### 19.8 插件兼容协议还可版本化

entry point 解决了发现问题，但还需要解决兼容问题：

- 插件 API 版本
- 主包版本范围
- 重复 key 冲突策略
- 插件能力声明
- 插件加载顺序
- 隔离测试

---

# 第八部分：诊断工具箱

## 20. 查“安装了什么”

```bash
python -m pip show lightx2v
python -m pip list
python -m pip check
```

`pip check` 用于检查已安装发行包的依赖约束是否互相满足。

查看 metadata：

```bash
python -c "from importlib.metadata import metadata; print(metadata('lightx2v'))"
```

查看版本：

```bash
python -c "from importlib.metadata import version; print(version('lightx2v'))"
```

### 20.1 查“导入了哪一份”

```bash
python -c "import lightx2v; print(lightx2v.__file__)"
python -c "import importlib.util; print(importlib.util.find_spec('lightx2v'))"
```

如果路径不是预期工作区，通常是：

- 环境里还有旧版本
- 当前目录中有同名文件遮蔽包
- 可编辑安装指向另一份 checkout
- `PYTHONPATH` 插入了其他路径

### 20.2 查依赖树

可以使用：

```bash
python -m pip install pipdeptree
python -m pipdeptree -p lightx2v
```

重点排查：

- 谁引入了冲突版本
- torch 相关包版本是否一致
- 同一库是否被多个顶层依赖施加冲突约束

### 20.3 查 import 耗时

```bash
python -X importtime -c "import lightx2v" 2> importtime.log
```

也可以使用 `tuna` 可视化 import time。

这对定位顶层导入过重、可选依赖扫描过慢很有帮助。

### 20.4 查 entry point

```bash
python -c "from importlib.metadata import entry_points; print(list(entry_points(group='lightx2v.platform_plugins')))"
```

如果插件包已安装但列表为空，应检查：

- 插件 `pyproject.toml` 的 group 名
- 是否重新安装了修改 metadata 后的插件
- 当前 Python 是否是安装插件的同一个环境
- wheel 的 `.dist-info/entry_points.txt`

### 20.5 查 wheel

```bash
python -m build
python -m zipfile -l dist/*.whl
python -m twine check dist/*
```

本地扩展还应检查：

```bash
python -c "import lightx2v_kernel; print(lightx2v_kernel.__file__)"
```

Linux 下可进一步用 `ldd` 检查 `.so` 动态链接依赖。

### 20.6 查环境可复现性

```bash
python -m pip freeze
python -m pip debug --verbose
```

`pip debug --verbose` 可以查看当前解释器支持的 wheel tags，解释“为什么这个 wheel 不兼容”。

---

## 21. 常见故障的系统排查法

### 21.1 `ModuleNotFoundError: No module named 'lightx2v'`

按顺序检查：

1. 当前 Python：`sys.executable`
2. 是否安装：`python -m pip show lightx2v`
3. `sys.path`
4. 是否在正确虚拟环境
5. 是否只在另一份 Python 中安装
6. wheel 是否实际包含 `lightx2v`

### 21.2 文件存在但 Runner 未注册

检查：

1. 对应模块是否被入口导入
2. 装饰器是否执行
3. 注册 key 是否与 `model_cls` 完全一致
4. 是否导入了另一份已安装源码
5. 模块导入是否在装饰器前因可选依赖失败

可临时检查：

```python
from lightx2v.utils.registry_factory import RUNNER_REGISTER

print(sorted(RUNNER_REGISTER.keys()))
```

### 21.3 修改源码不生效

检查：

- `module.__file__`
- 是否普通安装而非 editable 安装
- 当前进程是否已经缓存模块
- 是否修改了错误 checkout
- 是否修改的是生成文件或另一份扩展源码
- 本地扩展是否需要重新编译

### 21.4 安装成功但 kernel 导入失败

安装器所说的“成功”只表示 wheel 安装完成，不代表动态库运行时一定完整。

继续检查：

- wheel 平台标签
- Python ABI
- torch 版本
- CUDA/SYCL runtime
- GPU 架构
- 动态库搜索路径
- 编译时和运行时 C++ ABI

### 21.5 配置选择 Flash Attention 后才失败

这是延迟可选依赖模式的正常表现。检查：

- 依赖是否安装
- 导入符号是否与版本匹配
- 当前硬件是否支持
- 配置 key 是否映射到正确实现
- 实现是否在注册器中

---

# 第九部分：如何为新模型或新后端设计包边界

## 22. 新模型通常应放在哪里

若模型与主框架版本强耦合，并复用大量内部基类，放入主包更简单：

```text
lightx2v/models/runners/my_model/
lightx2v/models/networks/my_model/
lightx2v/models/schedulers/my_model/
```

需要同步处理：

- 模块 import
- Runner 注册
- 配置文件
- 输入输出类型
- 可选依赖
- wheel 中的 package data
- CI smoke test

### 22.1 什么时候考虑独立发行包

适合独立插件的情况：

- 厂商硬件后端
- 闭源算子
- 发布节奏不同
- 依赖巨大且用户群有限
- 许可证不同
- 需要独立 wheel 矩阵

但独立包会增加：

- API 兼容成本
- 跨仓库调试成本
- 版本组合数量
- 发布和 CI 复杂度

不能为了“模块化”而无条件拆包。

## 23. 新平台插件清单

1. 定义独立发行包名与导入包名
2. 声明兼容的 Python 和 LightX2V 版本
3. 定义 `lightx2v.platform_plugins` entry point
4. 实现无参数注册函数
5. 注册 Device 和平台算子
6. 避免 import 时初始化大资源
7. 处理第三方动态库缺失
8. 构建目标平台 wheel
9. 在干净环境安装 wheel
10. 验证 entry point 可发现
11. 验证 `PLATFORM=<name> import lightx2v`
12. 验证最小算子和模型推理
13. 检查重复 key 和加载顺序
14. 发布兼容矩阵

## 24. 新可选功能依赖清单

新增一个可选功能时，不要直接把所有依赖塞进基础列表。先回答：

- 普通用户是否必需？
- 是否只在某个模型或后端中 import？
- 能否延迟导入？
- 应属于哪个 extra？
- 未安装时错误是否清晰？
- 依赖是否有平台 wheel？
- 是否与 torch/CUDA 强耦合？
- CI 是否覆盖安装和未安装两种路径？

---

# 第十部分：一个更理想的 LightX2V 包管理蓝图

## 25. 推荐的发行包拆分

以下是设计参考，不代表当前仓库已经如此实现：

```text
lightx2v-core
  ├─ 核心 Pipeline、Runner、网络抽象
  └─ 最小通用依赖

lightx2v[server]
  ├─ FastAPI、uvicorn
  ├─ Redis、数据库
  └─ 对象存储

lightx2v[train]
  ├─ 训练器
  ├─ 监控
  └─ 训练数据工具

lightx2v[cuda]
  └─ CUDA 推荐算子集合

lightx2v-kernel
  └─ 自维护 CUDA kernel wheel

sycl-kernels
  └─ Intel XPU kernel wheel

vendor-lightx2v-plugin
  └─ 厂商平台插件
```

真正拆分时还需解决内部 API 稳定性，不能只移动目录。

## 26. 推荐的 CI 包管理流水线

```text
静态检查
  → 构建 sdist 和 wheel
  → twine check
  → 检查 wheel 文件清单
  → 在干净环境安装 wheel
  → 从仓库外执行 import smoke test
  → 验证 console scripts
  → 验证 extras 可分别安装
  → pip check
  → 平台 kernel wheel smoke test
  → entry point 插件发现测试
  → 最小模型/算子测试
```

至少应覆盖：

- Python 3.10、3.11、3.12
- 核心无可选 kernel 环境
- 一个标准 CUDA 环境
- 支持的平台环境
- editable 和 wheel 两种安装模式

## 27. 推荐的版本策略

```text
Git tag
   │
   ├─ 生成主包版本
   ├─ 生成文档版本
   └─ 生成兼容矩阵
```

避免在多个文件手工维护版本。

如果 kernel 与主包独立发布，应维护兼容关系：

```text
lightx2v 0.2.x
  ├─ lightx2v-kernel >=0.3,<0.4
  └─ plugin API v1
```

---

# 第十一部分：动手实验

## 28. 实验一：观察导入路径

在仓库内和仓库外分别执行：

```bash
python -c "import sys, lightx2v; print(sys.executable); print(lightx2v.__file__)"
```

回答：

1. 两次导入的是同一份源码吗？
2. 删除 editable 安装后，仓库外还能导入吗？
3. 当前目录对结果有什么影响？

## 29. 实验二：观察模块缓存

```python
import sys
import lightx2v

first = id(lightx2v)
import lightx2v
second = id(lightx2v)

print(first == second)
print("lightx2v" in sys.modules)
```

思考：为什么注册装饰器不会在第二次 import 时再次执行？

## 30. 实验三：构建并检查 wheel

```bash
python -m build
python -m zipfile -l dist/*.whl
```

检查：

- `lightx2v_platform` 是否存在
- `lightx2v_train` 是否被打包
- 配置 JSON 是否存在
- 测试和文档是否被误带
- wheel 文件大小是否合理

## 31. 实验四：在干净环境验证

```bash
python -m venv /tmp/lightx2v-wheel-test
source /tmp/lightx2v-wheel-test/bin/activate
python -m pip install dist/*.whl
cd /tmp
python -c "import lightx2v; print(lightx2v.__file__)"
python -m pip check
```

GPU 框架的完整安装可能较重，实际执行前应根据平台准备正确 torch 和系统运行时。

## 32. 实验五：查看平台插件

```bash
python -c "from importlib.metadata import entry_points; print(list(entry_points(group='lightx2v.platform_plugins')))"
```

如果没有安装外部插件，返回空列表是正常的。

## 33. 实验六：比较依赖清单

手工或编写脚本比较：

```text
pyproject.toml [project].dependencies
requirements.txt
```

将差异分为：

- 构建依赖
- 核心运行依赖
- 可选后端依赖
- Server 依赖
- 训练依赖
- 文档依赖
- 平台专属依赖

然后设计一份 extras 拆分方案。

---

# 第十二部分：常见误区

## 34. 误区总结

### 误区一：“仓库目录名就是 pip 包名”

错误。项目名、发行包名和导入包名可以不同。

### 误区二：“装上发行包后，仓库里所有目录都会被安装”

错误。具体由构建后端和 package discovery 配置决定。

### 误区三：“有 `__init__.py` 就一定会进入 wheel”

错误。还要满足包发现和制品包含规则。

### 误区四：“没有 `__init__.py` 就一定不能 import”

错误。Python 3 支持隐式命名空间包。

### 误区五：“`pip install -e .` 不需要重新安装”

错误。修改 metadata、entry point、依赖或扩展构建配置后通常需要重装。

### 误区六：“requirements.txt 就是标准发行 metadata”

错误。现代发行 metadata 主要来自 `pyproject.toml`；requirements 更适合环境或场景清单。

### 误区七：“版本范围等于可复现”

错误。范围允许解析结果随时间变化；部署需要 lock 或 constraints。

### 误区八：“可选依赖 import 失败就应该让整个框架退出”

不一定。未选择的后端应尽量不阻塞核心功能，但实际使用时必须明确失败。

### 误区九：“注册器会自动扫描整个目录”

错误。LightX2V 的装饰器注册依赖模块实际被 import。

### 误区十：“entry point 与装饰器注册器是同一个东西”

错误。entry point 是发行包 metadata 级发现；装饰器是进程内对象注册。

### 误区十一：“主包和 kernel 放在同一仓库就会一起安装”

错误。LightX2V 的 kernel 有独立 `pyproject.toml` 和构建后端。

### 误区十二：“安装成功说明 GPU 扩展一定可运行”

错误。动态库、驱动、ABI 和硬件架构仍可能不兼容。

---

# 第十三部分：源码索引

## 35. 核心文件

| 主题 | 文件 |
|---|---|
| 主发行包 metadata | `LightX2V/pyproject.toml:1-119` |
| 主包公共 API | `LightX2V/lightx2v/__init__.py:1-18` |
| 平台初始化和插件发现 | `LightX2V/lightx2v_platform/set_ai_device.py:1-53` |
| 平台设备初始化 | `LightX2V/lightx2v_platform/base/base.py:1-36` |
| 平台注册器 | `LightX2V/lightx2v_platform/registry_factory.py:1-73` |
| 框架注册器与平台 merge | `LightX2V/lightx2v/utils/registry_factory.py:1-88` |
| CLI 的 Runner 导入与注册触发 | `LightX2V/lightx2v/infer.py:1-57` |
| Server 模块入口 | `LightX2V/lightx2v/server/__main__.py:1-32` |
| 可选 Flash Attention 导入 | `LightX2V/lightx2v/common/ops/attn/flash_attn.py:1-29` |
| CUDA kernel 构建配置 | `LightX2V/lightx2v_kernel/pyproject.toml:1-40` |
| XPU kernel 构建配置 | `LightX2V/lightx2v_kernel_xpu/pyproject.toml:1-29` |
| 源码安装说明 | `LightX2V/README_zh.md:151-169` |
| 可编辑安装与可选算子 | `LightX2V/docs/ZH_CN/source/getting_started/quickstart.md:72-141` |

---

# 第十四部分：思考题

## 36. 思考题

1. 为什么 `pip show lightx2v` 显示的是发行包，而 `lightx2v.__file__` 显示的是导入模块路径？二者可能不一致吗？
2. 为什么在 `sys.modules` 中提前放入模块对象有助于递归导入，却仍不能自动解决循环导入？
3. LightX2V 在 `import lightx2v` 时初始化平台有什么好处和代价？如果改成显式 `initialize()`，哪些调用点需要变化？
4. 为什么 Runner 装饰器不能替代 entry point？entry point 又为什么不能直接替代运行时注册器？
5. 根项目的 `include = ["lightx2v*"]` 可能把哪些目录打入 wheel？这是否符合预期？
6. 为什么 CUDA kernel 应独立构建 wheel，而不是简单放入主包目录？
7. 为什么库 metadata 适合版本范围，而生产部署仍需要 constraints 或 lock？
8. 如果 `flash_attn` 未安装，框架应该在 import 时失败、创建 Pipeline 时失败，还是选择该算子时失败？分别有什么优缺点？
9. 如何让 `pyproject.toml` 和多个 requirements 文件不再漂移？
10. 如果第三方插件注册了与主框架相同的 attention key，应该覆盖、拒绝还是按优先级处理？如何保证可诊断性？
11. 如果将 `lightx2v_train` 拆成独立发行包，需要稳定哪些内部 API？
12. 如何设计 CI，证明 editable 安装可运行不只是因为仓库根目录碰巧在 `sys.path`？

---

## 37. 本课总结

Python 包管理可以分成三条链：

```text
源码与 metadata
  → 构建系统生成发行制品
  → 安装器解析依赖并写入环境
  → import system 在运行时加载模块
```

LightX2V 展示了大型 AI 框架中几种重要实践：

- 用 `pyproject.toml` 管理现代 Python 构建
- 将主包与 CUDA、XPU kernel wheel 分离
- 用平台抽象隔离厂商差异
- 用 entry point 发现外部平台插件
- 用装饰器注册器完成进程内路由
- 对可选高性能算子采用延迟依赖处理
- 用顶层 facade 暴露简洁 API

同时，它也提醒我们：

- 自动包发现必须验证实际 wheel 内容
- 基础依赖应控制规模并按 extras 拆分
- 多份依赖清单需要单一事实源
- 版本范围不等于可复现环境
- import 副作用和加载顺序会影响运行时能力
- 插件发现之后还需要兼容协议

真正优雅的包管理，不是把所有依赖写进一个文件，而是让**源码边界、发行边界、运行时扩展边界和用户安装场景保持一致**。
