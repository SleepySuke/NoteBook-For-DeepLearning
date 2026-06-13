# DL - d2l 深度学习学习记录

这个仓库用于记录我学习《动手学深度学习》的过程。当前主要学习网站：

[https://zh.d2l.ai/](https://zh.d2l.ai/)

目前学习进度在 **预备知识**，所以现阶段的代码和笔记主要放在 `datas/` 目录中。后续章节已经先按 d2l 的主章节创建好目录，目录名不带章节编号，方便后面直接补代码和笔记。

## 当前内容

```text
.
├── README.md
├── pyproject.toml
├── datas/                    # 预备知识
│   ├── README.md
│   ├── torch_1.py             # 数据操作
│   └── torch_2.py             # 数据预处理
├── linear_neural_networks/    # 线性神经网络
├── multilayer_perceptrons/    # 多层感知机
├── deep_learning_computation/ # 深度学习计算
├── convolutional_neural_networks/
├── modern_convolutional_neural_networks/
├── recurrent_neural_networks/
├── modern_recurrent_neural_networks/
├── attention_mechanisms/
├── optimization_algorithms/
├── computational_performance/
├── computer_vision/
├── nlp_pretraining/
└── nlp_applications/
```

没有创建第 16 章“附录：深度学习工具”对应目录；目前这个仓库先聚焦模型、训练和应用相关章节。

## 章节目录

| d2l 章节 | 本仓库目录 | 状态 |
| --- | --- | --- |
| 预备知识 | `datas/` | 学习中 |
| 线性神经网络 | `linear_neural_networks/` | 待学习 |
| 多层感知机 | `multilayer_perceptrons/` | 待学习 |
| 深度学习计算 | `deep_learning_computation/` | 待学习 |
| 卷积神经网络 | `convolutional_neural_networks/` | 待学习 |
| 现代卷积神经网络 | `modern_convolutional_neural_networks/` | 待学习 |
| 循环神经网络 | `recurrent_neural_networks/` | 待学习 |
| 现代循环神经网络 | `modern_recurrent_neural_networks/` | 待学习 |
| 注意力机制 | `attention_mechanisms/` | 待学习 |
| 优化算法 | `optimization_algorithms/` | 待学习 |
| 计算性能 | `computational_performance/` | 待学习 |
| 计算机视觉 | `computer_vision/` | 待学习 |
| 自然语言处理：预训练 | `nlp_pretraining/` | 待学习 |
| 自然语言处理：应用 | `nlp_applications/` | 待学习 |

## 学习重点

当前的重点是把深度学习所需的基础操作跑通：

- PyTorch 张量创建、形状变换、索引和切片
- 张量逐元素运算、矩阵运算、广播机制
- 布尔张量和掩码筛选
- 使用 Pandas 读取和处理表格数据
- 缺失值处理、类别变量独热编码
- 将预处理后的数据转换为 PyTorch Tensor

## 运行环境

当前项目主要使用 `uv` 管理 Python 环境和依赖。`pyproject.toml` 中已经声明了项目依赖。

### uv

同步环境：

```bash
uv sync
```

运行数据操作练习：

```bash
uv run python datas/torch_1.py
```

运行数据预处理练习：

```bash
cd datas
uv run python torch_2.py
cd ..
```

说明：`torch_2.py` 会使用相对路径创建示例 CSV 文件，因此建议在 `datas/` 目录下运行。

检查 PyTorch 是否可用：

```bash
uv run python -c "import torch; print(torch.__version__)"
```

如果后续需要补充绘图、Notebook 或视觉相关依赖，可以按需添加：

```bash
uv add matplotlib jupyter torchvision torchaudio
```

### conda

如果使用 `conda`：

```bash
conda create -n dl python=3.12
conda activate dl
python -m pip install -U pip
python -m pip install torch torchvision torchaudio numpy pandas
python datas/torch_1.py
```

运行 `torch_2.py` 时建议进入 `datas/`：

```bash
cd datas
python torch_2.py
cd ..
```

如果需要 GPU 版本的 PyTorch，请根据自己的系统、显卡和 CUDA 版本选择 PyTorch 官方安装命令。

### venv + pip

如果只使用 Python 自带虚拟环境：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install torch torchvision torchaudio numpy pandas
python datas/torch_1.py
```

运行 `torch_2.py`：

```bash
cd datas
python torch_2.py
cd ..
```

退出环境：

```bash
deactivate
```

### 直接使用系统 Python

不推荐长期把依赖安装到系统 Python，但临时验证可以这样做：

```bash
python -m pip install --user torch torchvision torchaudio numpy pandas
python datas/torch_1.py
```

## 学习记录方式

每次学习建议至少留下三类内容：

1. 可运行代码：用最小例子验证概念。
2. 关键笔记：记录概念、公式、API 和容易混淆的点。
3. 运行复盘：记录输出结果、报错原因和自己的理解。

现阶段可以先保持简单：一个主题一个 `.py` 文件，相关说明写到 `datas/README.md`。

## 参考资料

- [《动手学深度学习》中文版](https://zh.d2l.ai/)
- [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/)
- [PyTorch 官方文档](https://pytorch.org/docs/stable/)
- [uv 官方文档](https://docs.astral.sh/uv/)
