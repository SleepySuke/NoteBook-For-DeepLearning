# -*- coding: UTF-8 -*-
'''
@Author ：suke
@Version ：1.0
@Date ：2026-06-13
@Description：
deep learning 数据预处理（d2l.ai 2.2 节）
'''
import os
import pandas as pd
import torch

# ====== 1. 创建人工数据集 ======
print("===== 1. 读取数据集 =====\n")

# 创建 data 目录
os.makedirs(os.path.join('..', 'datas', 'data'), exist_ok=True)
data_file = os.path.join('..', 'datas', 'data', 'house_tiny.csv')

# 手动写入 CSV：NumRooms(房间数), Alley(巷子类型), Price(价格)
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')      # 列名
    f.write('NA,Pave,127500\n')            # 样本1: 房间数缺失, 巷子=Pave
    f.write('2,NA,106000\n')               # 样本2: 巷子缺失
    f.write('4,NA,178100\n')               # 样本3: 巷子缺失
    f.write('NA,NA,140000\n')              # 样本4: 房间数和巷子都缺失

print(f"CSV 文件已创建: {data_file}")

# 用 pandas 读取 CSV
data = pd.read_csv(data_file)
print("原始数据:\n", data)
#    NumRooms Alley   Price
# 0       NaN  Pave  127500
# 1       2.0   NaN  106000
# 2       4.0   NaN  178100
# 3       NaN   NaN  140000

# ====== 2. 处理缺失值 ======
print("\n===== 2. 处理缺失值 =====\n")

# 分离输入（前两列）和输出（最后一列）
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
print("inputs (前两列):")
print(inputs)
print("\noutputs (最后一列):")
print(outputs)

# 2a. 数值列缺失值：用该列的均值填充
# NumRooms 列: NaN, 2, 4, NaN → 均值 = (2+4)/2 = 3.0
print("\n--- 数值列: fillna(mean) ---")
print("NumRooms 均值:", inputs['NumRooms'].mean())
inputs = inputs.fillna(inputs.mean(numeric_only=True))
# 新版 pandas: mean() 需要 numeric_only=True，否则字符串列会报错
print("填充后:\n", inputs)
#    NumRooms Alley
# 0       3.0  Pave
# 1       2.0   NaN
# 2       4.0   NaN
# 3       3.0   NaN

# 2b. 类别列缺失值：独热编码（One-Hot Encoding）
# Alley 列的值: "Pave" 和 NaN → 编码为 Alley_Pave 和 Alley_nan 两列
print("\n--- 类别列: get_dummies(dummy_na=True) ---")
# 注意: get_dummies 默认返回 bool 类型，.astype(float) 转为数值方便后续转张量
inputs = pd.get_dummies(inputs, dummy_na=True).astype(float)
print("独热编码后:\n", inputs)
#    NumRooms  Alley_Pave  Alley_nan
# 0       3.0      True      False
# 1       2.0     False       True
# 2       4.0     False       True
# 3       3.0     False       True

print("\n类型分布:")
print(inputs.dtypes)

# ====== 3. 转换为张量格式 ======
print("\n===== 3. 转换为张量格式 =====\n")

# pandas DataFrame → numpy → torch tensor
X = torch.tensor(inputs.to_numpy(dtype=float), dtype=torch.float32)
y = torch.tensor(outputs.to_numpy(dtype=float), dtype=torch.float32)
print("X (输入张量):\n", X)
print("形状:", X.shape)
print("\ny (输出张量):\n", y)
print("形状:", y.shape)

# ====== 4. 观察与理解 ======
print("\n===== 4. 观察与理解 =====\n")

# 每列的含义
print("X 的列: NumRooms(房间数), Alley_Pave(是否Pave), Alley_nan(是否缺失巷子)")
print("y 的列: Price(价格)")

# 检查是否有 NaN 残留
print("\nX 中有 NaN?", torch.isnan(X).any().item())
print("y 中有 NaN?", torch.isnan(y).any().item())

# 样本数量 vs 特征数量
print(f"\n样本数: {X.shape[0]}, 特征数: {X.shape[1]}")

# ====== 5. 练习 ======
print("\n===== 5. 课后练习 =====")

# 练习1: 创建包含更多行和列的原始数据集
print("\n--- 练习1: 更大的数据集 ---")

# 创建一个 6 行 5 列的数据集
big_file = os.path.join('..', 'datas', 'data', 'big_house.csv')
with open(big_file, 'w') as f:
    f.write('Rooms,Area,Year,Type,Price\n')
    f.write('3,120,2000,Apartment,300000\n')
    f.write('2,NA,1995,House,250000\n')
    f.write('NA,80,2010,NA,200000\n')
    f.write('5,200,NA,Apartment,500000\n')
    f.write('4,150,2005,House,NA\n')
    f.write('NA,NA,NA,NA,350000\n')

big_data = pd.read_csv(big_file)
print("原始数据 (6×5):\n", big_data)
print("\n每列缺失值数量:\n", big_data.isna().sum())

# 分离输入和输出
big_inputs = big_data.iloc[:, 0:4]
big_outputs = big_data.iloc[:, 4]

# 数值列填充均值
numeric_cols = ['Rooms', 'Area', 'Year']
for col in numeric_cols:
    if col in big_inputs.columns:
        big_inputs[col] = big_inputs[col].fillna(big_inputs[col].mean())

# 类别列独热编码
big_inputs = pd.get_dummies(big_inputs, dummy_na=True).astype(float)
print("\n预处理后:\n", big_inputs)
print("特征列:", list(big_inputs.columns))

# 转张量
X_big = torch.tensor(big_inputs.to_numpy(dtype=float), dtype=torch.float32)
y_big = torch.tensor(big_outputs.to_numpy(dtype=float), dtype=torch.float32)
print("\nX_big 形状:", X_big.shape)
print("y_big 形状:", y_big.shape)

# 练习2: 删除缺失值最多的列
print("\n--- 练习2: 删除缺失值最多的列 ---")

# 重新读取原始数据
data2 = pd.read_csv(big_file)
print("原始每列缺失数:\n", data2.isna().sum())

# 找到缺失值最多的列并删除
max_nan_col = data2.isna().sum().idxmax()
max_nan_count = data2.isna().sum().max()
print(f"\n缺失最多: '{max_nan_col}' ({max_nan_count} 个缺失值)")

data2_dropped = data2.drop(columns=[max_nan_col])
print("\n删除后:\n", data2_dropped)
print("剩余列:", list(data2_dropped.columns))

# 练习3: 预处理后转张量
print("\n--- 练习3: 预处理后转张量 ---")

# 对新数据集完整走一遍预处理流程
inputs3 = data2_dropped.iloc[:, 0:3]    # 前3列是输入
outputs3 = data2_dropped.iloc[:, 3]     # 第4列(原第5列)是输出

print("处理前 inputs3 缺失值:\n", inputs3.isna().sum())

# 填充数值列的缺失值
for col in inputs3.columns:
    if inputs3[col].dtype in ['float64', 'int64']:
        inputs3[col] = inputs3[col].fillna(inputs3[col].mean())

# 独热编码
inputs3 = pd.get_dummies(inputs3, dummy_na=True).astype(float)

X3 = torch.tensor(inputs3.to_numpy(dtype=float), dtype=torch.float32)
y3 = torch.tensor(outputs3.to_numpy(dtype=float), dtype=torch.float32)

print("\n最终 X3:\n", X3)
print("X3 形状:", X3.shape)
print("y3:", y3)

print("\n===== 完成 =====")
print("数据预处理的完整流程: CSV读取 → 分离输入输出 → 数值填充均值 → 类别独热编码 → 转张量")
