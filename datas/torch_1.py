# -*- coding: UTF-8 -*-
'''
@Author ：suke
@Version ：1.0
@Date ：2026-05-13 22:42:32
@Description：
deep learning 数据操作
'''
import torch

# 创建一个包含12个元素的张量，元素值为0到11
x = torch.arange(12)
print(x)

# 通过张量的shape属性访问张量的形状
print(x.shape)

# 张量的size方法也可以访问张量的形状
print(x.size())

# 张量元素的总数
total_elements = x.numel()
print(total_elements)

# 将张量重塑为3行4列的二维张量
x_reshaped = x.reshape(3, 4)
print(x_reshaped)

x_reshaped_again = x.reshape(-1, 4)
print(x_reshaped_again)

x_reshaped_again2 = x.reshape(3, -1)
print(x_reshaped_again2)

# 设置全零张量
zeros_tensor = torch.zeros(2,3,4)
print(zeros_tensor)

# 设置全一张量
ones_tensor = torch.ones(2,3,4)
print(ones_tensor)

# 随机初始化张量
random_tensor = torch.rand(2,3,4)
print(random_tensor)

# python列表转张量
list_data = [[1, 2, 3], [4, 5, 6]]
tensor_from_list = torch.tensor(list_data)
print(tensor_from_list)

# 运算符操作
# 注意：以下运算都是逐元素（element-wise）的，即每个位置独立计算

print("\n===== 运算符操作 =====\n")

a = torch.tensor([1.0, 2.0, 3.0, 4.0])
b = torch.tensor([2.0, 2.0, 2.0, 2.0])


print("exp", torch.exp(a))  # e^a_i
print("log", torch.log(a))  # ln(a_i)

# 逐元素加减乘除
print("a + b =", a + b)    # [3, 4, 5, 6]  对应位置相加
print("a - b =", a - b)    # [-1, 0, 1, 2] 对应位置相减
print("a * b =", a * b)    # [2, 4, 6, 8]  对应位置相乘（不是矩阵乘法！）
print("a / b =", a / b)    # [0.5, 1, 1.5, 2]
print("a ** 2 =", a ** 2)  # [1, 4, 9, 16] 每个元素平方

# 广播机制：形状不同时，PyTorch 自动扩展
print("\na * 10 =", a * 10)              # 标量 10 自动广播为 [10,10,10,10]
m = torch.tensor([[1.0], [2.0], [3.0]])  # 形状 (3, 1)
n = torch.tensor([10.0, 20.0])            # 形状 (2,)
print("广播加法结果 (3,1) + (2,):\n", m + n)
# (3,1) 广播成 (3,2)，(2,) 广播成 (3,2)，然后相加

# 矩阵乘法（真正的矩阵乘法，不是逐元素相乘）
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # (2,2)
y = torch.tensor([[5.0, 6.0], [7.0, 8.0]])  # (2,2)
print("\n矩阵乘法 x @ y:\n", x @ y)           # @ 运算符 = 矩阵乘法
print("等价于 torch.matmul(x, y):\n", torch.matmul(x, y))

# ====== 比较运算符：== / > / < / != ======
# 比较运算逐元素执行，返回布尔张量 (dtype=torch.bool)
print("\n===== 比较运算符 =====\n")

a = torch.tensor([1.0, 2.0, 3.0, 4.0])
b = torch.tensor([1.0, 0.0, 3.0, 5.0])

print("a:", a)
print("b:", b)

# 逐元素比较
print("\na == b:", a == b)          # [True, False, True, False]
print("a != b:", a != b)          # [False, True, False, True]
print("a > b :", a > b)           # [False, True, False, False]
print("a < b :", a < b)           # [False, False, False, True]

# 统计满足条件的元素个数
print("\n相等元素数 (a == b).sum():", (a == b).sum().item())   # 2
print("a > b 的元素数:",           (a > b).sum().item())        # 1

# 所有/任意 判断
print("\n全部相等？torch.all(a == b):", torch.all(a == b).item())    # False (用 torch.all)
print("存在相等？torch.any(a == b):", torch.any(a == b).item())    # True

# ====== 用 == 验证计算结果 ======
print("\n===== 用 == 验证计算结果 =====\n")

# 验证拼接结果
X = torch.arange(6, dtype=torch.float32).reshape(2, 3)
Y = torch.tensor([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
Z_cat = torch.cat((X, Y), dim=0)  # 期望: (4, 3)
print("X:\n", X)
print("Y:\n", Y)
print("cat dim=0:\n", Z_cat)
# 验证 Z_cat 的前两行 == X
print("\n前2行应 == X:", torch.equal(Z_cat[:2], X))   # True
# 验证 Z_cat 的后两行 == Y
print("后2行应 == Y:", torch.equal(Z_cat[2:], Y))   # True

# 验证广播结果
A = torch.tensor([[1.0], [2.0], [3.0]])  # (3, 1)
B = torch.tensor([10.0, 20.0])            # (2,)
C = A + B                                 # (3, 2)
print("\nA (3,1):\n", A)
print("B (2,):\n", B)
print("A + B:\n", C)

# 验证第一列: 每行 = A 对应行 + 10
col0_check = C[:, 0] == (A.squeeze() + B[0])
print("第0列 == A行+10:", col0_check)   # [True, True, True]

# 验证第二列: 每行 = A 对应行 + 20
col1_check = C[:, 1] == (A.squeeze() + B[1])
print("第1列 == A行+20:", col1_check)   # [True, True, True]

# 逐个位置验证
print("\nC[0,0] == 1+10:", C[0, 0] == (1.0 + 10.0))   # True
print("C[2,1] == 3+20:", C[2, 1] == (3.0 + 20.0))     # True

# ====== 广播 × 比较：批量验证异常值 ======
print("\n===== 广播 + 比较：批量阈值检测 =====\n")

# 4 个样本，每个样本 3 个维度的数据
data = torch.tensor([
    [0.5, 1.2, 3.0],
    [2.0, 0.8, 1.5],
    [3.5, 2.0, 0.1],
    [1.0, 2.5, 2.2]
])  # (4, 3)
threshold = torch.tensor([1.0, 1.5, 2.0])  # (3,)
print("数据 (4,3):\n", data)
print("阈值 (3,):\n", threshold)

# 广播比较：每行与 threshold 逐列比较
exceed = data > threshold   # (4, 3) 布尔张量
print("\n每列是否超过阈值:\n", exceed)

# 统计每列超标数
print("\n每列超标数:", exceed.sum(dim=0))   # dim=0 按列求和

# 筛选"所有维度都超标"的样本
all_exceed = torch.all(exceed, dim=1)  # dim=1 按行检查
print("所有维度都超标的行:", all_exceed)  # [False, False, False, True]

# 用布尔掩码提取满足条件的行
print("筛选出的样本:\n", data[all_exceed])   # 第4行 [1.0, 2.5, 2.2]

# ====== 2D 广播比较：生成任意形状的布尔张量 ======
print("\n===== 2D 广播比较：生成布尔掩码 =====\n")

# 比较运算符 + 广播 = 任意形状的布尔张量
X = torch.tensor([[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 10, 11, 12]])        # (3, 4)
threshold_row = torch.tensor([3, 6, 9, 12])   # (4,)

print("X (3,4):\n", X)
print("阈值 (4,):", threshold_row)
print("\nX > threshold_row  → 广播为 (3,4) 布尔张量:")
mask = X > threshold_row
print(mask)
# tensor([[False, False, False, False],
#         [ True, False, False, False],
#         [ True,  True,  True, False]])

print("\ndtype:", mask.dtype)       # torch.bool
print("True 的数量:", mask.sum().item())  # 4

# 布尔掩码的两种用法
# 用法1: 直接当索引 → 取出 True 位置的元素
print("\nX[mask] 取出超标元素:", X[mask])   # [5, 9, 10, 11]

# 用法2: 赋值 — 超标的值全部设为 -1
X_copy = X.clone()
X_copy[mask] = -1
print("超标位置设 -1 后:\n", X_copy)

# ====== 行列维度上的布尔运算 ======
print("\n===== 行列维度上的布尔运算 =====\n")

M = torch.tensor([[1, 2, 3],
                   [2, 3, 4],
                   [3, 4, 5]])  # (3, 3)
print("M:\n", M)

# 每行是否有偶数
row_has_even = torch.any(M % 2 == 0, dim=1)  # dim=1 压缩列，返回 (3,)
print("每行有偶数？", row_has_even)            # [True, True, True]

# 每列是否全大于 1
col_all_gt1 = torch.all(M > 1, dim=0)         # dim=0 压缩行，返回 (3,)
print("每列全 > 1？", col_all_gt1)             # [False, True, True]

# 两矩阵逐元素相等 → 2D 布尔张量
A = torch.tensor([[1, 2], [3, 4]])
B = torch.tensor([[1, 0], [3, 5]])
print("\nA:\n", A)
print("B:\n", B)
print("A == B (2D 布尔张量):\n", A == B)
# tensor([[ True, False],
#         [ True, False]])

# ====== 3D 广播比较 ======
print("\n===== 3D 广播比较 =====\n")

# 3D 与 1D 广播：规则完全一样
T3d = torch.tensor([[[1, 2], [3, 4]],
                     [[5, 6], [7, 8]]])          # (2, 2, 2)
threshold_1d = torch.tensor([3, 5])               # (2,)
print("T3d (2,2,2):\n", T3d)
print("阈值 (2,):", threshold_1d)

# 右对齐: (2,2,2) vs (2,) → (2,2,2) vs (1,1,2) → 广播 dim=0/1
mask_3d = T3d > threshold_1d
print("\nT3d > threshold  (2,2,2) 布尔张量:")
print(mask_3d)
print("dtype:", mask_3d.dtype)

# 3D 布尔张量同样可以直接索引
print("\nT3d[mask_3d] 提取超标元素:", T3d[mask_3d])

# 3D 也可以沿指定 dim 归约
# dim=2 压缩最内层 → 每行（最后一维）是否全超阈值
print("\ntorch.all(mask_3d, dim=2):")  # 沿最内层判断
print(torch.all(mask_3d, dim=2))

# ====== 1D 对比：dim 受限 ======
print("\n===== 1D 比较与 dim 限制 =====\n")

v = torch.tensor([1, 2, 3, 4, 5])
print("v:", v)
print("v > 2:", v > 2)                       # 1D 布尔张量 (5,)
print("(v > 2).sum():", (v > 2).sum().item())  # 3

# 1D 只有 dim=0，dim=1 会报错
print("\ntorch.any(v > 2) →", torch.any(v > 2).item())   # any() 不传 dim → 全局
print("torch.all(v > 0) →", torch.all(v > 0).item())

try:
    torch.any(v > 2, dim=1)  # 1D 没有 dim=1!
except IndexError as e:
    print("torch.any(v>2, dim=1) 报错:", str(e))

# ====== torch.equal vs == vs torch.allclose ======
print("\n===== torch.equal / == / allclose 对比 =====\n")

T1 = torch.tensor([0.1 + 0.2])                 # 浮点 0.30000000000000004
T2 = torch.tensor([0.3])                       # 0.3
print("0.1+0.2:", T1.item())
print("0.3:    ", T2.item())

print("\n== 比较:", T1 == T2)                   # False (浮点精度导致不相等)
print("torch.equal:", torch.equal(T1, T2))      # False
print("torch.allclose:", torch.allclose(T1, T2))  # True (带容差，推荐用于浮点验证)

# 指数和对数
# exp(x) = e^x，其中 e ≈ 2.71828（自然常数）
# exp(1)=2.718, exp(2)=7.389, exp(3)=20.086
print("\nexp(a):", torch.exp(a))   # e^ai
print("log(a):", torch.log(a))     # ln(ai)

# 张量拼接：dim=0 上下堆 vs dim=1 左右拼
print("\n===== 张量拼接 =====\n")
X = torch.arange(12, dtype=torch.float32).reshape((3, 4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print("X:\n", X)
print("Y:\n", Y)
print("\ndim=0 (上下堆，行数+行数，形状 6×4):\n", torch.cat((X, Y), dim=0))
print("形状:", torch.cat((X, Y), dim=0).shape)
print("\ndim=1 (左右拼，列数+列数，形状 3×8):\n", torch.cat((X, Y), dim=1))
print("形状:", torch.cat((X, Y), dim=1).shape)

# 索引与切片：和 Python 列表一样，左闭右开
print("\n===== 索引与切片 =====\n")
X = torch.arange(12).reshape(3, 4)
print("原始 X:\n", X)
print("X[-1] (最后一行):", X[-1])          # 取最后一行
print("X[1:3] (第1~2行，左闭右开):\n", X[1:3])
print("X[:, 1] (第1列):", X[:, 1])         # : 表示取所有行
# 赋值：把前两行的第2、3列设为 999
X[0:2, 1:3] = 999
print("赋值 X[0:2, 1:3] = 999 后:\n", X)

# 节省内存：原地操作避免反复分配新内存
print("\n===== 节省内存 =====\n")
X = torch.arange(12).reshape(3, 4)
print("修改前 X 的内存地址:", id(X))
X = X + 1                              # 这会在新地址创建张量
print("X = X + 1 后地址:", id(X), "(变了！新张量)")
X[:] = X + 1                           # 切片赋值 = 原地写入
print("X[:] = X + 1 后地址:", id(X), "(不变，原地更新)")
X += 1                                 # += 也是原地操作
print("X += 1 后地址:", id(X), "(不变)")

# Tensor 与 NumPy 互转（共享底层内存）
print("\n===== Tensor ↔ NumPy =====\n")
X = torch.arange(12, dtype=torch.float32).reshape(3, 4)
A = X.numpy()                              # Tensor → NumPy
print("NumPy 数组:\n", A)
B = torch.tensor(A)                        # NumPy → Tensor
print("转回 Tensor:\n", B)
# 共享内存：修改 tensor 会影响 numpy，反之亦然
X[0, 0] = 999.0
print("修改 X[0,0]=999 后，NumPy A 也变了:\n", A)  # A 的 [0,0] 也变成 999

# 单元素张量转 Python 标量
a = torch.tensor([3.5])
print("\n.item() 转为 Python float:", a.item())

# 广播规则深度演示
print("\n===== 广播规则详解 =====\n")
# 核心：对齐形状 → 从右往左逐维比较 → 每维要么相等，要么有一方为 1

# ✅ 案例1: (3, 1) + (2,) — 1维可扩
A = torch.arange(3, dtype=torch.float32).reshape(3, 1)  # [[0],[1],[2]]
B = torch.tensor([10.0, 20.0])                            # [10, 20]
print("A (3,1):\n", A)
print("B (2,):\n", B)
print("A + B (广播为3×2):\n", A + B)

# ✅ 案例2: (4,) + (3, 4) — 标量扩成高维
A = torch.tensor([1.0, 2.0, 3.0, 4.0])  # (4,)
B = torch.ones(3, 4)       
C = A + B                 # (3, 4)
print("(4,) + (3,4) 结果形状:", (A + B).shape)  # A 扩成 (3,4)
print("结果:\n", C)

# ✅ 案例3: (3, 1, 1) + (1, 4, 2) — 多维度广播
A = torch.ones(3, 1, 1)
B = torch.ones(1, 4, 2)
print("A (3,1,1):\n", A)
print("B (1,4,2):\n", B)
print("(3,1,1)+(1,4,2) 结果形状:", (A + B).shape)  # (3, 4, 2)

# ❌ 案例4: (3, 3) + (1, 2) — dim=1: 3 vs 2 互不相等也不为1
try:
    A = torch.ones(3, 3)
    B = torch.ones(1, 2)
    A + B
except RuntimeError as e:
    print("(3,3)+(1,2) 报错:", str(e)[:60])

# 广播判断过程可视化
print("\n===== 广播判断过程可视化 =====\n")


def can_broadcast(shape_a, shape_b):
    """模拟 PyTorch 内部的广播判断逻辑"""
    # 1. 右对齐，短的那个前面补 1
    a = list(shape_a)
    b = list(shape_b)
    while len(a) < len(b):
        a.insert(0, 1)
    while len(b) < len(a):
        b.insert(0, 1)

    result = []
    for i, (sa, sb) in enumerate(zip(a, b)):
        if sa == sb:
            result.append(sa)
            print(f"  dim={i}: {sa} vs {sb} → 相等 → 取 {sa}")
        elif sa == 1:
            result.append(sb)
            print(f"  dim={i}: {sa} vs {sb} → 一方为1 → 广播成 {sb}")
        elif sb == 1:
            result.append(sa)
            print(f"  dim={i}: {sa} vs {sb} → 一方为1 → 广播成 {sa}")
        else:
            print(f"  dim={i}: {sa} vs {sb} → 都不为1 ❌ 直接报错")
            return None
    return tuple(result)


print("检查 (3, 1) + (2,):")
r = can_broadcast((3, 1), (2,))
print(f"  结果形状: {r}\n")

print("检查 (3, 3) + (1, 2):")
r = can_broadcast((3, 3), (1, 2))
print(f"  结果: {r}\n")

print("检查 (3, 1, 1) + (1, 4, 2):")
r = can_broadcast((3, 1, 1), (1, 4, 2))
print(f"  结果形状: {r}")

# 广播扩展过程可视化
print("\n===== (3,1) + (2,) 扩展过程 =====\n")

A = torch.tensor([[1.0], [2.0], [3.0]])  # (3, 1)
B = torch.tensor([10.0, 20.0])            # (2,)
C = A + B                                 # (3, 2)

print("原始 A (3,1):")
print(A)
print("\n原始 B (2,):")
print(B)

# 用 expand 展示逻辑上的扩展（不实际复制数据）
A_exp = A.expand(3, 2)  # 将 dim=1 从 1 扩成 2
B_exp = B.expand(3, 2)  # 将 dim=0 从 1(补齐后) 扩成 3
print("\nA 逻辑扩展为 (3,2) —— 每列复制：")
print(A_exp)
print("\nB 逻辑扩展为 (3,2) —— 每行复制：")
print(B_exp)
print("\n逐元素相加 = 结果：")
print(C)

# 证明 expand 不复制数据
print(f"\nA_exp 与 A 共享内存？{A_exp.storage().data_ptr() == A.storage().data_ptr()}  (是！只是改变了 view)")


# 练习
X = torch.arange(12, dtype=torch.float32).reshape((3,4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print("X:\n", X)
print("Y:\n", Y)
print('X==Y', X == Y)
print('X<Y的结果', X < Y)
print('X>Y的结果', X > Y)
print('X<=Y的结果', X <= Y)
print('X>=Y的结果', X >= Y)

# ====== 练习：3D 广播 + 验证 ======
print("\n===== 练习：3D 广播 — 结果是否与预期相同？ =====\n")

A = torch.tensor([[[1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0]]])  # (2, 3, 1)
B = torch.tensor([10.0, 20.0])                                        # (2,)

print("A (2,3,1):")
print(A)
print("\nB (2,):", B)

# 广播加法
C = A + B  # 预期: (2,3,2)
print("\nA + B 广播结果 (2,3,2):")
print(C)
print(f"形状: {C.shape}  (预期: (2,3,2))")

# 验证1: 形状是否正确
assert C.shape == (2, 3, 2), f"形状错误: {C.shape}"

# 验证2: A 的 dim=2 被扩成 2 列，每列复制相同值
# 即 C[batch, row, :] 的两列应该 = A[batch, row, 0] + B[:]
for batch in range(2):
    for row in range(3):
        expected = A[batch, row, 0] + B  # (2,)
        actual = C[batch, row, :]         # (2,)
        match = torch.equal(actual, expected)
        print(f"  C[{batch},{row},:] == A[{batch},{row},0] + B? {match}")
        assert match, "值不符合广播预期！"

# 验证3: 比较运算符同样适用于 3D 广播
print("\nC > 15 (3D 布尔掩码):")
print(C > 15)
print("超标元素:", C[C > 15])

print("\n✅ 3D 广播结果与预期完全一致！")
print("结论：无论几维，广播规则（右对齐 → 逐维比较 → 只扩1）始终生效。")