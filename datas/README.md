# datas - 预备知识

预备知识相关内容`datas/` 下的所有脚本都默认属于预备知识阶段。

## 当前文件

| 文件 | 内容 |
| --- | --- |
| `torch_1.py` | PyTorch 数据操作练习 |
| `torch_2.py` | 数据预处理练习 |

## `torch_1.py`：数据操作

这个文件主要练习 PyTorch Tensor 的基础操作。

已经覆盖的内容包括：

- 使用 `torch.arange` 创建一维张量
- 查看张量形状：`shape`、`size`
- 统计元素个数：`numel`
- 张量形状变换：`reshape`
- 创建全 0、全 1、随机张量
- Python 列表转换为张量
- 逐元素运算：加、减、乘、除、幂运算
- 常见数学函数：`exp`、`log`
- 广播机制
- 矩阵乘法：`@` 和 `torch.matmul`
- 比较运算：`==`、`!=`、`>`、`<`
- 布尔张量、布尔掩码和条件筛选
- `torch.all`、`torch.any`、按维度统计布尔结果

运行方式：

```bash
uv run python datas/torch_1.py
```

## `torch_2.py`：数据预处理

这个文件主要练习从原始表格数据到 Tensor 的完整预处理流程。

已经覆盖的内容包括：

- 使用 Python 创建简单 CSV 数据集
- 使用 Pandas 读取 CSV 文件
- 区分输入特征和输出标签
- 处理数值列缺失值：用均值填充
- 处理类别列缺失值：使用独热编码
- 使用 `pd.get_dummies` 转换类别变量
- 将 Pandas DataFrame / Series 转换为 PyTorch Tensor
- 检查 Tensor 中是否仍然存在 NaN
- 构造更大的练习数据集
- 统计每列缺失值数量
- 删除缺失值最多的列
- 重新完成预处理并转换为 Tensor

建议运行方式：

```bash
cd datas
uv run python torch_2.py
cd ..
```

说明：`torch_2.py` 中使用了相对路径创建 `data/` 目录和示例 CSV 文件，所以建议在 `datas/` 目录下运行。

运行后可能生成：

```text
datas/data/
├── house_tiny.csv
└── big_house.csv
```

这些文件是学习数据预处理时生成的临时小数据集。

## 当前阶段需要掌握的点

预备知识阶段的目标不是训练复杂模型，而是熟悉后续章节会频繁用到的基本工具：

- 能看懂 Tensor 的形状变化
- 能判断一个运算是逐元素运算、广播运算还是矩阵运算
- 能用布尔掩码筛选数据
- 能把表格数据处理成模型可以接收的数值张量
- 能理解缺失值、类别变量和数值变量的基本处理方法
- 能用代码验证自己对 API 行为的理解

## 后续可以继续补充

后续预备知识还可以继续在当前目录下添加新的脚本，例如：

```text
datas/
├── torch_1.py
├── torch_2.py
├── linear_algebra.py
├── calculus.py
├── autograd.py
└── probability.py
```

也可以继续沿用当前命名方式，例如 `torch_3.py`、`torch_4.py`。关键是每个文件聚焦一个主题，能直接运行，方便之后复习。
