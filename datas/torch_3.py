# -*- coding: UTF-8 -*-
'''
@Author ：suke
@Version ：1.0
@Date ：2026-06-14
@Description：
deep learning 线性代数
'''
import torch

# ====== 1. 标量 ======
print("===== 1. 标量（Scalar）=====\n")

x = torch.tensor(3.0)
y = torch.tensor(2.0)

print(f"x = {x}")
print(f"y = {y}")
print(f"x + y = {x + y}")
print(f"x * y = {x * y}")
print(f"x / y = {x / y}")
print(f"x ** y = {x ** y}")
print(f"x 是标量？{x.dim() == 0}，维度数 = {x.dim()}")

# ====== 2. 向量 ======
print("\n===== 2. 向量（Vector）=====\n")

x = torch.arange(4)
print(f"x = {x}")
print(f"len(x) = {len(x)}")           # 长度
print(f"x.shape = {x.shape}")         # 形状
print(f"x.size() = {x.size()}")       # 等价写法
print(f"x.dim() = {x.dim()}")         # 维度数（轴数）= 1
print(f"x[3] = {x[3]}")               # 索引第 4 个元素
print(f"x[-1] = {x[-1]}")             # 最后一个元素

# 向量 = 一维张量
print(f"\n向量也是张量: isinstance(x, torch.Tensor) = {isinstance(x, torch.Tensor)}")

# ====== 3. 矩阵 ======
print("\n===== 3. 矩阵（Matrix）=====\n")

A = torch.arange(20).reshape(5, 4)
print(f"A (5×4):\n{A}")
print(f"A.shape = {A.shape}")
print(f"A.dim() = {A.dim()}  (2 个轴)")

# 访问元素: A[i, j]
print(f"\nA[2, 3] = {A[2, 3].item()}  (第3行第4列)")
print(f"A[1:3] (第1~2行):\n{A[1:3]}")
print(f"A[:, 1] (第1列): {A[:, 1]}")

# 转置
print(f"\nA.T (转置, 4×5):\n{A.T}")
print(f"A.T.shape = {A.T.shape}")

# 对称矩阵：B == B.T
B = torch.tensor([[1, 2, 3],
                   [2, 0, 4],
                   [3, 4, 5]])
print(f"\nB:\n{B}")
print(f"B == B.T? (是否对称):\n{B == B.T}")
print(f"完全对称？{torch.equal(B, B.T)}")

# ====== 4. 张量 ======
print("\n===== 4. 张量（Tensor）=====\n")

X = torch.arange(24).reshape(2, 3, 4)
print(f"X (2×3×4):\n{X}")
print(f"X.shape = {X.shape}")
print(f"X.dim() = {X.dim()}")
print(f"X.numel() = {X.numel()}  (总元素数 = 2×3×4)")

# ====== 5. 张量算法的基本性质 ======
print("\n===== 5. 张量算法的基本性质 =====\n")

A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone()  # 复制一份，后续修改不影响原矩阵

print(f"A:\n{A}")
print(f"\nB:\n{B}")

# 5a. Hadamard 积（逐元素乘法）
print(f"\n--- 5a. Hadamard 积: A * B ---")
print(f"A * B:\n{A * B}")
print("注意: 这不是矩阵乘法！对应位置直接相乘。")

# 5b. 标量与张量的运算
print(f"\n--- 5b. 标量运算 ---")
print(f"A * 2 + 1:\n{A * 2 + 1}")
print("标量自动广播到每个元素，形状不变。")

# ====== 6. 降维 ======
print("\n===== 6. 降维（Reduction）=====\n")

A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
print(f"A (5×4):\n{A}")

# 6a. 求和
print(f"\n--- 6a. sum() ---")
print(f"A.sum() = {A.sum():.1f}  → 所有元素求和，返回标量")
print(f"A.sum(axis=0) = {A.sum(axis=0)}  → 沿 axis=0 求和, 形状 {A.sum(axis=0).shape}")
print("含义: 每列的和（把 5 行压成 1 行）")
print(f"A.sum(axis=1) = {A.sum(axis=1)}  → 沿 axis=1 求和, 形状 {A.sum(axis=1).shape}")
print("含义: 每行的和（把 4 列压成 1 列）")

# 6b. 均值
print(f"\n--- 6b. mean() ---")
print(f"A.mean() = {A.mean():.1f}")
print(f"A.mean(axis=0) = {A.mean(axis=0)}")
print(f"A.mean(axis=1) = {A.mean(axis=1)}")

# 6c. 非降维求和: keepdim=True
print(f"\n--- 6c. keepdim=True ---")
sum_A = A.sum(axis=1, keepdim=True)
print(f"A.sum(axis=1) 形状: {A.sum(axis=1).shape}      ← 轴消失了")
print(f"A.sum(axis=1, keepdim=True) 形状: {sum_A.shape}  ← 轴保留为1")
print(f"\nsum_A:\n{sum_A}")

# 关键用法: 每行除以自己的和 → 行归一化
print(f"\nA / sum_A (每行归一化):\n{A / sum_A}")
print("每行的元素之和 = 1.0")

# 验证: 每行和应该全是 1
row_sums = (A / sum_A).sum(axis=1)
print(f"验证每行和 == 1.0: {row_sums}")

# 6d. 沿多个轴求和
print(f"\n--- 6d. 多轴降维 ---")
X = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
print(f"X (2×3×4) → sum(axis=0) 形状: {X.sum(axis=0).shape}")  # (3, 4)
print(f"X (2×3×4) → sum(axis=1) 形状: {X.sum(axis=1).shape}")  # (2, 4)
print(f"X (2×3×4) → sum(axis=2) 形状: {X.sum(axis=2).shape}")  # (2, 3)
print(f"X (2×3×4) → sum(axis=[0,1]) 形状: {X.sum(axis=(0,1)).shape}")  # (4,)
print(f"X (2×3×4) → sum(axis=[0,1,2]) 形状: {X.sum(axis=(0,1,2)).shape}")  # 标量

# ====== 7. 点积 ======
print("\n===== 7. 点积（Dot Product）=====\n")

x = torch.ones(4, dtype=torch.float32)
y = torch.tensor([1.0, 2.0, 3.0, 4.0])

print(f"x = {x}")
print(f"y = {y}")

# 三种等价写法
dot1 = torch.dot(x, y)
dot2 = torch.sum(x * y)
dot3 = (x * y).sum()

print(f"\ntorch.dot(x, y)       = {dot1:.1f}")
print(f"torch.sum(x * y)      = {dot2:.1f}")
print(f"(x * y).sum()         = {dot3:.1f}")
print(f"手工计算: 1*1+1*2+1*3+1*4 = {1*1 + 1*2 + 1*3 + 1*4:.1f}")

# 几何意义: x·y = ||x|| × ||y|| × cos(θ)
x_norm = torch.norm(x)
y_norm = torch.norm(y)
cos_theta = dot1 / (x_norm * y_norm)
print(f"\ncos(θ) = (x·y) / (||x|| × ||y||) = {dot1:.1f} / ({x_norm:.1f} × {y_norm:.2f}) = {cos_theta:.4f}")
print("cos(θ) 越接近 1 → 方向越一致; 接近 0 → 正交; 接近 -1 → 方向相反")

# ====== 8. 矩阵-向量积 ======
print("\n===== 8. 矩阵-向量积（Matrix-Vector Product）=====\n")

A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
x = torch.arange(4, dtype=torch.float32)

print(f"A (5×4):\n{A}")
print(f"\nx (4,): {x}")
print(f"\ntorch.mv(A, x) = {torch.mv(A, x)}")
print(f"结果形状: {torch.mv(A, x).shape}")

# 手工验证第 i 个结果 = A[i,:] 与 x 的点积
print("\n手工验证每一行:")
for i in range(5):
    manual = torch.dot(A[i], x)
    mv_result = torch.mv(A, x)[i]
    print(f"  第{i}行: A[{i}]·x = {manual:.1f}, torch.mv = {mv_result:.1f}, 匹配: {torch.allclose(manual, mv_result)}")

# ====== 9. 矩阵-矩阵乘法 ======
print("\n===== 9. 矩阵-矩阵乘法（Matrix-Matrix Multiplication）=====\n")

A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = torch.ones(4, 3)

print(f"A (5×4):\n{A}")
print(f"\nB (4×3):\n{B}")

# 三种等价写法
C1 = A @ B
C2 = torch.matmul(A, B)
C3 = torch.mm(A, B)

print(f"\nC = A @ B (5×3):\n{C1}")
print(f"C.shape = {C1.shape}")
print(f"A@B == matmul: {(C1 == C2).all().item()}")
print(f"A@B == mm:     {(C1 == C3).all().item()}")

# 手工验证 c[2,1] = A[2,:] · B[:,1]
print(f"\n手工验证 C[2,1]:")
print(f"  A[2,:] = {A[2]}")
print(f"  B[:,1] = {B[:,1]}")
print(f"  点积 = {torch.dot(A[2], B[:,1]):.1f}")
print(f"  C[2,1] = {C1[2, 1]:.1f}")

# 模拟一层全连接网络
print(f"\n--- 模拟全连接层 ---")
batch = torch.randn(64, 784)     # 64 个样本, 每样本 784 维
W1 = torch.randn(784, 256)       # 输入 784 → 隐藏 256
W2 = torch.randn(256, 10)        # 隐藏 256 → 输出 10

hidden = batch @ W1               # (64, 784) @ (784, 256) = (64, 256)
output = hidden @ W2              # (64, 256) @ (256, 10) = (64, 10)

print(f"batch (64, 784) @ W1 (784, 256) → hidden {hidden.shape}")
print(f"hidden (64, 256) @ W2 (256, 10) → output {output.shape}")
print("两次矩阵乘法: 784维 → 256维 → 10分类输出")

# ====== 10. 范数 ======
print("\n===== 10. 范数（Norm）=====\n")

# 10a. L2 范数
print("--- 10a. L2 范数 ---")

u = torch.tensor([3.0, 4.0])
l2_torch = torch.norm(u)
l2_manual = torch.sqrt(torch.sum(u ** 2))

print(f"u = {u}")
print(f"torch.norm(u)          = {l2_torch:.1f}")
print(f"sqrt(sum(u²)) 手工验证  = {l2_manual:.1f}")
print(f"勾股定理: √(3²+4²) = √25 = 5.0")
print(f"匹配: {torch.allclose(l2_torch, l2_manual)}")

# 10b. L1 范数
print(f"\n--- 10b. L1 范数 ---")
l1_torch = torch.norm(u, p=1)
l1_manual = torch.abs(u).sum()

print(f"torch.norm(u, p=1)     = {l1_torch:.1f}")
print(f"torch.abs(u).sum() 手工 = {l1_manual:.1f}")
print("L1 = |3| + |4| = 7 (曼哈顿距离)")

# 10c. L2 vs L1 对大值的敏感度
print(f"\n--- 10c. L2 vs L1 对大值的敏感度 ---")

v1 = torch.tensor([10.0, 0.0])
v2 = torch.tensor([5.0, 5.0])

print(f"v1 = {v1}, v2 = {v2}")
print(f"L1(v1) = {torch.norm(v1, p=1):.1f}, L1(v2) = {torch.norm(v2, p=1):.1f}")
print(f"  → L1 下两者长度一致（曼哈顿距离相等）")
print(f"L2(v1) = {torch.norm(v1):.2f}, L2(v2) = {torch.norm(v2):.2f}")
print(f"  → L2 下 v1 > v2（大值被平方放大）")

# 10d. Frobenius 范数
print(f"\n--- 10d. Frobenius 范数 ---")

A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
fro_torch = torch.norm(A, 'fro')
fro_manual = torch.sqrt(torch.sum(A ** 2))

print(f"A:\n{A}")
print(f"torch.norm(A, 'fro')        = {fro_torch:.4f}")
print(f"sqrt(sum(A²)) 手工验证       = {fro_manual:.4f}")
print("就是把矩阵拉平成一个向量，再算 L2。")

# 10e. 各范数对比
print(f"\n--- 10e. 各范数对比 ---")

vec = torch.tensor([1.0, -2.0, 3.0, -4.0])
print(f"向量: {vec}")
print(f"L1 范数   (|x_i| 求和)  = {torch.norm(vec, p=1):.1f}")
print(f"L2 范数   (sqrt(Σ|x|²)) = {torch.norm(vec, p=2):.4f}")
print(f"L∞ 范数   (max|x_i|)    = {torch.norm(vec, p=float('inf')):.1f}")
print(f"元素绝对值之和 (手动L1)    = {torch.abs(vec).sum():.1f}")
print(f"元素平方和开根号 (手动L2)   = {torch.sqrt((vec ** 2).sum()):.4f}")

# ====== 11. 小结与练习 ======
print("\n===== 11. 小结与练习 =====\n")

print("核心知识清单:")
print("  1. 标量(0D)→向量(1D)→矩阵(2D)→张量(3D+) 是数据的组织形式")
print("  2. 矩阵乘法(A@B) 是整个前向传播的数学基础")
print("  3. sum/mean 沿轴降维时, axis 参数决定哪个轴被'压扁'")
print("  4. Hadamard积(A*B) ≠ 矩阵乘法(A@B), 前者逐元素, 后者行列点积")
print("  5. L2范数量误差 → MSE loss; L1范数量误差 → MAE loss")
print("  6. L2 对大值敏感(平方放大), L1 对小值和大值一视同仁(线性)")

# 练习1: 证明 (A^T)^T = A
print(f"\n--- 练习1: 证明 (Aᵀ)ᵀ = A ---")
A = torch.arange(6, dtype=torch.float32).reshape(2, 3)
A_T_T = A.T.T
print(f"A:\n{A}")
print(f"(A.T).T:\n{A_T_T}")
print(f"(Aᵀ)ᵀ == A? {torch.equal(A, A_T_T)}")

# 练习2: 证明 A^T + B^T = (A + B)^T
print(f"\n--- 练习2: 证明 Aᵀ + Bᵀ = (A + B)ᵀ ---")
A = torch.arange(6, dtype=torch.float32).reshape(2, 3)
B = torch.ones(2, 3) * 10
left = A.T + B.T
right = (A + B).T
print(f"left (Aᵀ + Bᵀ):\n{left}")
print(f"right ((A+B)ᵀ):\n{right}")
print(f"相等? {torch.equal(left, right)}")

# 练习3: 任意方阵 A, A + A^T 总是对称的吗？
print(f"\n--- 练习3: A + Aᵀ 总是对称的吗？ ---")
A = torch.tensor([[1.0, 2.0, 3.0],
                   [4.0, 5.0, 6.0],
                   [7.0, 8.0, 9.0]])
S = A + A.T
print(f"A:\n{A}")
print(f"S = A + Aᵀ:\n{S}")
print(f"S 对称? (S == Sᵀ):\n{S == S.T}")
print(f"完全对称? {torch.equal(S, S.T)}")
print("原因: (A+Aᵀ)ᵀ = Aᵀ + (Aᵀ)ᵀ = Aᵀ + A = A + Aᵀ, 所以永远对称。")

# 练习4: len(X) 在张量上返回什么？
print(f"\n--- 练习4: len(X) 的含义 ---")
X = torch.arange(24).reshape(2, 3, 4)
print(f"X.shape = {X.shape}")
print(f"len(X) = {len(X)}")
print("len(X) 始终返回 axis=0 的长度（第一个维度）。")

# 练习5: A / A.sum(axis=1) 会发生什么？
print(f"\n--- 练习5: A / A.sum(axis=1) 的广播 ---")
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
try:
    result = A / A.sum(axis=1)
    print("运行成功:")
    print(result)
    print("注意: (5,4) / (5,) → (5,) 广播成 (5,1) → (5,4), 结果每行被除的是它的行和。")
except RuntimeError as e:
    print(f"报错: {e}")
    print("因为 (5,4) 和 (5,) 广播时, 右对齐为 (5,4) 和 (1,5) → dim=1: 4 vs 5 不兼容!")

# 练习6: 不同轴上的求和形状
print(f"\n--- 练习6: (2,3,4) 张量各轴求和形状 ---")
X = torch.ones(2, 3, 4)
print(f"原始形状: {X.shape}")
print(f"sum(axis=0): {X.sum(axis=0).shape}  (压缩第0轴)")
print(f"sum(axis=1): {X.sum(axis=1).shape}  (压缩第1轴)")
print(f"sum(axis=2): {X.sum(axis=2).shape}  (压缩第2轴)")

print(f"\n===== 完成 =====")
print("线性代数总结: 数据是张量, 变换是矩阵乘法, 目标是范数。")
