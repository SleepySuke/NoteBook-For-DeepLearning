# -*- coding: UTF-8 -*-
'''
@Author ：suke
@Version ：1.0
@Date ：2026-06-20
@Description：
deep learning 微积分与自动求导（d2l.ai 2.4 & 2.5 节）
'''
import os
import torch
import matplotlib.pyplot as plt
import numpy as np


# ====== d2l 绘图辅助函数 ======
# 这些是 d2l 包中 plot 函数的简化实现，不依赖 d2l 包

def use_svg_display():
    """使用 svg 格式在 Jupyter 中显示绘图 (本地脚本跳过)"""
    pass

def set_figsize(figsize=(3.5, 2.5)):
    """设置 matplotlib 的图表大小"""
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize

def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """设置 matplotlib 的轴"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid(True, alpha=0.3)

def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(6, 4), axes=None):
    """绘制数据点。d2l.plot 的本地实现。"""
    if legend is None:
        legend = []
    set_figsize(figsize)
    axes = axes if axes else plt.gca()

    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1 or
                isinstance(X, list) and not hasattr(X[0], "__len__"))

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)


# ====== 2.4.1 导数和微分 ======
print("===== 2.4.1 导数和微分 =====\n")

def f(x):
    """f(x) = 3x² - 4x"""
    return 3 * x ** 2 - 4 * x

print("函数 f(x) = 3x² - 4x")
print("精确导数 f'(x) = 6x - 4\n")

# 数值导数：用极限定义逼近，h 逐步缩小
def numerical_lim(f, x, h):
    return (f(x + h) - f(x)) / h

h = 0.1
print(f"在 x=1 处，不同 h 下的数值导数 (精确值: f'(1)=2):")
for i in range(5):
    num = numerical_lim(f, 1.0, h)
    print(f"  h={h:.5f}, numerical limit={num:.5f}, 误差={abs(num - 2):.5f}")
    h *= 0.1
print("h → 0 时，数值导数 → 精确值 2")

# 切线: 在 x=1 处 f(1)=-1, f'(1)=2 → 切线 y = 2(x-1)-1 = 2x-3
print(f"\n--- 切线可视化 ---")
print(f"在 x=1 处: f(1) = {f(1.0):.0f}, f'(1) = 2")
print(f"切线方程: y = 2x - 3")
print("(切线的斜率 = 导数, 切线经过切点 (1, -1))\n")

# d2l 风格: 用 plot 绘制 f(x) 和切线
x = np.arange(0, 3, 0.1)
fig, ax = plt.subplots(figsize=(6, 4))
plot(x, [f(x), 2 * x - 3], 'x', 'f(x)',
     legend=['f(x) = 3x² - 4x', 'Tangent line (x=1): y = 2x - 3'],
     axes=ax)
ax.annotate('(1, -1)', xy=(1, -1), xytext=(1.3, -2),
            arrowprops=dict(arrowstyle='->'), fontsize=10)

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, 'data')
os.makedirs(data_dir, exist_ok=True)
tangent_path = os.path.join(data_dir, 'calculus_tangent.png')
plt.savefig(tangent_path, dpi=120)
print(f"切线图已保存: {tangent_path}\n")

# d2l 中切线的概念验证: 对于 y = 2x - 3
# x=1 时 y=-1, 与 f(1)=-1 重合
# 切线 = 在切点处"无限逼近"曲线的直线
# 越靠近 x=1, 曲线和切线越接近
print("切线本质: 在切点附近, 曲线 ≈ 切线")
x_test = 1.0
h_vals = [0.5, 0.1, 0.01]
for h in h_vals:
    f_val = f(x_test + h)
    tangent_val = 2 * (x_test + h) - 3
    print(f"  x={x_test+h:.3f}: f(x)={f_val:.4f}, 切线={tangent_val:.2f}, 差距={abs(f_val-tangent_val):.4f}")
print("  h 越小 → 差距越小 → 切线是曲线的'局部线性近似'\n")


# ====== 求导规则：极限定义的推演 ======
print("===== 求导规则：从极限定义到速算公式 =====\n")
print("所有规则都从同一个定义推导: f'(x) = lim_{h→0} [f(x+h) - f(x)] / h\n")

h = 1e-5  # 用于数值验证的微小量

# --- 规则1: 常数规则 ---
print("--- 规则1: 常数规则 d/dx(c) = 0 ---")
c = 5.0
def const_f(x): return c
x0 = 3.0
num = (const_f(x0 + h) - const_f(x0)) / h
print(f"  f(x)=c={c},  x={x0}")
print(f"  极限推导: [f(x+h)-f(x)]/h = [{c}-{c}]/h = 0/h = 0  →  h→0 仍是 0")
print(f"  数值验证: ({c}-{c})/{h} = {num:.1f}")
print(f"  几何: 水平线的斜率为 0 — 常数函数不变化。\n")

# --- 规则2: 幂函数规则 ---
print("--- 规则2: 幂函数规则 d/dx(x^n) = n·x^(n-1) ---")

def show_power_rule(n, x0=2.0):
    """验证 x^n 的导数 = n·x^(n-1)"""
    def f_pow(x): return x ** n
    num = (f_pow(x0 + h) - f_pow(x0)) / h
    theory = n * x0 ** (n - 1)
    print(f"  f(x)=x^{n}, x={x0}")
    print(f"  极限推导: 二项式展开 (x+h)^n = x^n + n·x^(n-1)·h + ...")
    print(f"            [f(x+h)-f(x)]/h = [x^n + n·x^(n-1)·h + O(h²) - x^n]/h")
    print(f"                            = n·x^(n-1) + O(h)  →  h→0 时 = n·x^(n-1)")
    print(f"  公式: {n}·x^({n-1}) = {n}·{x0}^{n-1} = {theory}")
    print(f"  数值: {num:.6f}")
    return num, theory

show_power_rule(2)   # x² → 2x
print()
show_power_rule(3)   # x³ → 3x²
print()
show_power_rule(4)   # x⁴ → 4x³
print("  注意: 公式同样适用于负数/分数指数，如 d/dx(1/x) = d/dx(x^{-1}) = -1·x^{-2}\n")

# --- 规则3: 指数函数规则 ---
print("--- 规则3: 指数函数规则 d/dx(e^x) = e^x ---")
x0 = 1.0
num = (np.exp(x0 + h) - np.exp(x0)) / h
theory = np.exp(x0)
print(f"  f(x)=e^x, x={x0}")
print(f"  极限推导: [e^(x+h) - e^x]/h = e^x·[e^h - 1]/h")
print(f"  关键极限: lim(h→0) (e^h - 1)/h = 1")
print(f"            (因为 e^h = 1 + h + h²/2! + ..., (e^h-1)/h = 1 + h/2 + ... → 1)")
print(f"  因此: e^x·1 = e^x → 导函数 = 自身！")
print(f"  公式: e^{x0} = {theory:.6f}")
print(f"  数值: {num:.6f}")
print(f"  这是 e 的特殊性质: 唯一导函数等于自身的函数。\n")

# 一般指数: a^x
print(f"  推广 d/dx(a^x) = a^x·ln(a)")
a_val = 2.0
num = (a_val**(x0 + h) - a_val**x0) / h
theory = a_val**x0 * np.log(a_val)
print(f"  f(x)=2^x, x=1: 数值={num:.6f}, 公式=2^1·ln(2)={theory:.6f}\n")

# --- 规则4: 对数函数规则 ---
print("--- 规则4: 对数函数规则 d/dx(ln x) = 1/x ---")
x0 = 2.0
num = (np.log(x0 + h) - np.log(x0)) / h
theory = 1 / x0
print(f"  f(x)=ln(x), x={x0}")
print(f"  极限推导: [ln(x+h) - ln(x)]/h = [ln((x+h)/x)]/h = [ln(1 + h/x)]/h")
print(f"  令 t = h/x, 则 = [ln(1+t)]/(x·t) = (1/x)·[ln(1+t)/t]")
print(f"  关键极限: lim(t→0) ln(1+t)/t = 1")
print(f"  因此: (1/x)·1 = 1/x")
print(f"  公式: 1/{x0} = {theory:.6f}")
print(f"  数值: {num:.6f}\n")

# --- 规则5: 加法规则 (和的导数 = 导数的和) ---
print("--- 规则5: 加法规则 d/dx[f(x) + g(x)] = f'(x) + g'(x) ---")
def f1(x): return 3 * x**2
def f2(x): return -4 * x
x0 = 2.0
num = (f1(x0+h) + f2(x0+h) - f1(x0) - f2(x0)) / h
theory_f1 = 6 * x0   # f1' = 6x
theory_f2 = -4       # f2' = -4
print(f"  f(x)=3x²-4x, x={x0}")
print(f"  推导: 极限的线性性质 — 和的极限 = 极限的和")
print(f"        lim [(f(x+h)+g(x+h))-(f(x)+g(x))]/h")
print(f"        = lim [f(x+h)-f(x)]/h + lim [g(x+h)-g(x)]/h")
print(f"        = f'(x) + g'(x)")
print(f"  公式: 6x + (-4) = {theory_f1 + theory_f2}")
print(f"  数值: {num:.6f}\n")

# --- 规则6: 常数倍规则 ---
print("--- 规则6: 常数倍规则 d/dx[c·f(x)] = c·f'(x) ---")
c = 3.0
x0 = 2.0
num = (c * f1(x0+h) - c * f1(x0)) / h
theory = c * 6 * x0
print(f"  f(x)=3·(x²),  即 c=3, g(x)=x²")
print(f"  推导: [c·g(x+h) - c·g(x)]/h = c·[g(x+h)-g(x)]/h → c·g'(x)")
print(f"  公式: 3·2x = 3·{2*x0} = {theory}")
print(f"  数值: {num:.6f}\n")

# --- 规则7: 乘法规则 (积的导数) ---
print("--- 规则7: 乘法规则 d/dx[f(x)·g(x)] = f'(x)·g(x) + f(x)·g'(x) ---")
def g1(x): return x**2       # f
def g2(x): return np.sin(x)  # g → f'=2x, g'=cos(x)
x0 = 1.0
num = (g1(x0+h)*g2(x0+h) - g1(x0)*g2(x0)) / h
theory = (2*x0)*g2(x0) + g1(x0)*np.cos(x0)
print(f"  f(x)=x²·sin(x), x={x0}")
print(f"  推导: [f(x+h)g(x+h) - f(x)g(x)]/h")
print(f"        = [f(x+h)g(x+h) - f(x+h)g(x) + f(x+h)g(x) - f(x)g(x)]/h")
print(f"        = f(x+h)·[g(x+h)-g(x)]/h + g(x)·[f(x+h)-f(x)]/h")
print(f"        → f(x)·g'(x) + g(x)·f'(x)  (h→0)")
print(f"  公式: (2x)·sin(x) + x²·cos(x) = 2·sin(1)+1·cos(1) = {theory:.6f}")
print(f"  数值: {num:.6f}")
print(f"  记忆口诀: 前导后不导 + 前不导后导\n")

# --- 规则8: 除法规则 (商的导数) ---
print("--- 规则8: 除法规则 d/dx[f(x)/g(x)] = (f'g - fg')/g² ---")
def h1(x): return x**2       # f = x², f'=2x
def h2(x): return x + 1      # g = x+1, g'=1
x0 = 2.0
num = (h1(x0+h)/h2(x0+h) - h1(x0)/h2(x0)) / h
theory = (2*x0*(x0+1) - x0**2*1) / (x0+1)**2
print(f"  f(x)=x²/(x+1), x={x0}")
print(f"  推导: f/g = f·(1/g), 用乘法规则 + 链式法则")
print(f"        d/dx[f·g^{-1}] = f'·g^{-1} + f·(-1)g^{-2}·g'")
print(f"                       = (f'·g - f·g')/g²")
print(f"  公式: (2x·(x+1) - x²·1)/(x+1)² = {theory:.6f}")
print(f"  数值: {num:.6f}")
print(f"  记忆口诀: (上导下不导 - 上不导下导) 除以 下的平方\n")

# --- 规则9: 链式法则 (复合函数求导) ---
print("--- 规则9: 链式法则 d/dx[f(g(x))] = f'(g(x))·g'(x) ---")
def outer(x): return np.sin(x**2)   # 外: sin(u), 内: u=x²
x0 = 1.0
num = (outer(x0+h) - outer(x0)) / h
theory = np.cos(x0**2) * (2*x0)     # cos(u)·2x
print(f"  f(x)=sin(x²), x={x0}")
print(f"  推导: 令 u=g(x), y=f(u)")
print(f"        dy/dx = dy/du · du/dx = f'(g(x))·g'(x)")
print(f"        外函数 sin(u) 的导数 = cos(u), 内函数 x² 的导数 = 2x")
print(f"  公式: cos(x²)·2x = cos(1)·2 = {theory:.6f}")
print(f"  数值: {num:.6f}")
print(f"  记忆: 外导(留下内层不动) × 内导\n")

# --- 规则10: 三角函数规则 ---
print("--- 规则10: 三角函数规则 ---")
x0 = np.pi / 4
# sin 导数
num_sin = (np.sin(x0 + h) - np.sin(x0)) / h
theory_cos = np.cos(x0)
print(f"  d/dx(sin x) = cos x")
print(f"  推导: [sin(x+h)-sin(x)]/h")
print(f"        和角公式 sin(x+h)=sin x·cos h + cos x·sin h")
print(f"        = [sin x(cos h-1) + cos x·sin h]/h")
print(f"        = sin x·(cos h-1)/h + cos x·sin h/h")
print(f"        重要极限: sin h/h → 1, (cos h-1)/h → 0")
print(f"        → cos x")
print(f"  x=π/4: 数值={num_sin:.6f}, 理论={theory_cos:.6f}")

num_cos = (np.cos(x0 + h) - np.cos(x0)) / h
theory_neg_sin = -np.sin(x0)
print(f"\n  d/dx(cos x) = -sin x")
print(f"  类似推导，用 cos(x+h)=cos x·cos h - sin x·sin h")
print(f"  x=π/4: 数值={num_cos:.6f}, 理论={theory_neg_sin:.6f}\n")

# ====== 求导规则速查表 ======
print("=" * 68)
print("求导规则速查表")
print("=" * 68)
rules = [
    ("常数",   "d/dx c = 0",              "水平线斜率为 0"),
    ("幂函数", "d/dx x^n = n·x^(n-1)",   "指数搬下来, 指数减 1"),
    ("指数",   "d/dx e^x = e^x",          "导函数等于自身 (e 独有)"),
    ("一般指数","d/dx a^x = a^x·ln a",    "多乘一个 ln a"),
    ("对数",   "d/dx ln x = 1/x",         "倒数关系"),
    ("和差",   "d/dx (f±g) = f' ± g'",   "各自求导再相加"),
    ("常数倍", "d/dx (c·f) = c·f'",      "常数提到求导外"),
    ("积",     "d/dx (f·g) = f'g + fg'", "前导后不导 + 前不导后导"),
    ("商",     "d/dx (f/g) = (f'g-fg')/g²","分子导分母不导-分子不导分母导, 除以分母平方"),
    ("链式",   "dy/dx = dy/du · du/dx",  "外导(内不动) × 内导"),
    ("sin",    "d/dx sin x = cos x",      "正弦导数是余弦"),
    ("cos",    "d/dx cos x = -sin x",     "余弦导数是负正弦"),
]
for name, formula, mnemonic in rules:
    print(f"  {name:<8} {formula:<40}  {mnemonic}")
print()


# ====== 2.4.2 偏导数 ======
print("===== 2.4.2 偏导数 =====\n")

# 偏导数: 把其他变量当常数
# f(x₁, x₂) = 3x₁² + 5e^{x₂}  (练习2 用到)
# ∂f/∂x₁ = 6x₁,  ∂f/∂x₂ = 5e^{x₂}

x = torch.tensor([1.0, 2.0], requires_grad=True)
y = 3 * x[0]**2 + 5 * torch.exp(x[1])  # f = 3x₁² + 5e^{x₂}
y.backward()

print("f(x₁, x₂) = 3x₁² + 5e^{x₂} 在 (x₁=1, x₂=2) 处:")
print(f"  ∂f/∂x₁ = 6x₁ = 6×1 = 6")
print(f"  PyTorch: x.grad[0] = {x.grad[0]:.4f}")
print(f"  df/dx2 = 5*exp(x2) = 5*e^2 = {5*np.exp(2):.4f}")
print(f"  PyTorch: x.grad[1] = {x.grad[1]:.4f}")


# ====== 2.4.3 梯度 ======
print(f"\n===== 2.4.3 梯度 =====")

# 梯度 = 所有偏导组成的向量，指向上升最快的方向
# 练习3: f(x) = ||x||₂ 的梯度

x = torch.tensor([3.0, 4.0], requires_grad=True)
y = torch.norm(x, p=2)   # L2 范数
print(f"y = {y.item():.1f}"),y.backward()


print(f"\nf(x) = ||x||₂ = √(x₁² + x₂²)")
print(f"在 x = (3, 4) 处, ||x||₂ = {y.item():.1f}")
print(f"梯度 ∇||x||₂ = x / ||x||₂ = [{3/5}, {4/5}]")
print(f"PyTorch 梯度: {x.grad}")
print("含义: L2 范数的梯度是单位向量，指向远离原点的方向。")

# d2l 多元函数梯度规则
print(f"\n--- 多元函数梯度常用规则 (d2l 2.4.3) ---\n")
print("以下规则是深度学习中反复用到的基本功。\n")

# 规则G1: ∇_x (A x) = Aᵀ
print("--- 规则G1: ∇_x (A x) = A^T ---")
A = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])  # 3×2
x = torch.tensor([7., 8.], requires_grad=True)      # 2维
y = (A @ x).sum()   # A@x 是 3维向量, .sum() 转标量
y.backward()
print(f"  A (3×2) = [[1,2],[3,4],[5,6]]")
print(f"  x = (7, 8)")
print(f"  A@x = {(A @ x.detach()).tolist()}  (3维向量)")
print(f"  梯度: {x.grad.tolist()}")
print(f"  A^T 的每行求和的列向量: A^T·[1,1,1] = {A.T.sum(dim=1).tolist()}")
print(f"  推导: d(A@x)[k] / dx[j] = A[k,j]")
print(f"        对所有输出分量求和 -> sum_k A[k,j] = (A^T @ 1)[j]")
print(f"        更一般地，对于向量输出 y=Ax, dy/dx = A^T\n")

# 规则G2: ∇_x (x^T A) = A  (A ∈ R^{n×m})
print("--- 规则G2: ∇_x (x^T A) = A ---")
x = torch.tensor([1., 2., 3.], requires_grad=True)   # 3维
A = torch.tensor([[4., 5.], [6., 7.], [8., 9.]])    # 3×2
y_val = (x @ A).sum()
y_val.backward()
print(f"  x (3,) = (1, 2, 3)")
print(f"  A (3×2) = [[4,5],[6,7],[8,9]]")
print(f"  x^T A = {x.detach() @ A}  (2维行向量)")
print(f"  梯度: {x.grad.tolist()}")
print(f"  A 的每行求和: {A.sum(dim=1).tolist()}")
print(f"  推导: (x^T A)[j] = sum_i x_i * A[i,j]")
print(f"        d(x^T A)[j]/dx[i] = A[i,j]")
print(f"        对所有 j 求和 -> sum_j A[i,j] = (A @ 1)[i]\n")

# 规则G3: ∇_x (x^T A x) = (A + A^T)x  (二次型)
print("--- 规则G3: ∇_x (x^T A x) = (A + A^T)x ---")
x = torch.tensor([1., 2.], requires_grad=True)
A = torch.tensor([[3., 1.], [4., 5.]])   # 非对称矩阵
y_val = (x @ A @ x.T).sum()   # 二次型 x^T A x (标量)
# 注意: x^T A x 已经是标量，不需要 .sum()
y_val2 = x @ A @ x
y_val2.backward()
grad_theory = (A + A.T) @ x.detach()   # (A+A^T)x
print(f"  x = (1, 2)")
print(f"  A = [[3,1],[4,5]]  (非对称)")
print(f"  x^T A x = {y_val2.item():.0f}")
print(f"  梯度 (PyTorch):   {x.grad.tolist()}")
print(f"  (A+A^T)x (理论):  {grad_theory.tolist()}")
print(f"  推导思路: 把 x^T A x = Σᵢ Σⱼ xᵢ Aᵢⱼ xⱼ 展开逐项求导")
print(f"         ∂/∂x_k (x_i A_ij x_j)")
print(f"         = A_kj x_j (当 i=k) + x_i A_ik (当 j=k)")
print(f"         = (A x)_k + (A^T x)_k = ((A+A^T)x)_k")
print(f"  特例: 如果 A 对称(A=A^T)，则梯度 = 2Ax\n")

# 规则G4: ∇_x ||x||^2 = ∇_x (x^T x) = 2x
# (这是 G3 中 A=I 的特例)
print("--- 规则G4: ∇_x ||x||^2 = ∇_x (x^T x) = 2x ---")
x = torch.tensor([5., 3., 4.], requires_grad=True)
y_val = torch.dot(x, x)   # x^T x = ||x||²
y_val.backward()
print(f"  x = (5, 3, 4)")
print(f"  ||x||^2 = x·x = 25+9+16 = {y_val.item():.0f}")
print(f"  梯度: {x.grad.tolist()}")
print(f"  理论 2x: {(2 * x.detach()).tolist()}")
print(f"  这是梯度下降中最常用的公式之一。\n")

# 规则G5: ∇_X ||X||_F^2 = 2X
print("--- 规则G5: ∇_X ||X||_F^2 = 2X ---")
X = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)
f_val = torch.norm(X, 'fro') ** 2   # Frobenius 范数平方
f_val.backward()
print(f"  X = [[1,2],[3,4]]")
print(f"  ||X||_F^2 = 1+4+9+16 = {f_val.item():.0f}")
print(f"  梯度:\n{X.grad}")
print(f"  理论 2X:\n{2 * X.detach()}")
print(f"  推广: 矩阵的 Frobenius 范数平方梯度 = 2X, 与向量 L2 范数平方梯度 = 2x 形式一致。\n")

# 梯度规则速查表
print("=" * 68)
print("多元函数梯度规则速查表 (深度学习核心工具)")
print("=" * 68)
grad_rules = [
    ("线性变换(左乘)", "∇_x (A x) = A^T",            "A∈R^{m×n}, 输出是向量时用"),
    ("线性变换(右乘)", "∇_x (x^T A) = A",            "A∈R^{n×m}"),
    ("二次型",       "∇_x (x^T A x) = (A+A^T)x",   "A 对称时 = 2Ax, 非常重要!"),
    ("L2范数平方",    "∇_x ||x||_2^2 = 2x",         "二次型 A=I 的特例"),
    ("F范数平方",    "∇_X ||X||_F^2 = 2X",          "矩阵版的 L2 范数平方"),
    ("标量函数和",    "∇(f+g) = ∇f + ∇g",            "梯度也是线性的"),
    ("标量倍",       "∇(c·f) = c·∇f",               "常数提到梯度外"),
]
for name, formula, note in grad_rules:
    print(f"  {name:<16} {formula:<35} {note}")
print()

# 从偏导数到梯度到梯度下降的串联
print("--- 串联: 从偏导数到梯度下降 ---")
x = torch.tensor([5.0, 3.0], requires_grad=True)  # 从 (5, 3) 出发
lr = 0.1
print(f"目标: 最小化 f(x) = ||x||^2 = x1^2 + x2^2")
print(f"起点: x = (5, 3), f = 25+9 = 34")
print(f"梯度: ∇f = 2x, 负梯度 = -2x → 指向原点!\n")
print(f"{'步数':<6} {'x1':<10} {'x2':<10} {'||x||^2':<12} {'梯度':<20}")
for t in range(8):
    f_val = torch.dot(x, x)
    f_val.backward()
    with torch.no_grad():
        grad = x.grad.clone()
        x -= lr * grad   # 梯度下降
    print(f"{t:<6} {x[0].item():<10.4f} {x[1].item():<10.4f} {f_val.item():<12.4f} {grad.tolist()}")
    x.grad.zero_()
print(f"\n  x 从 (5,3) → (0,0), 每一步都沿负梯度 [2x₁, 2x₂] 方向走")
print("  梯度总指向函数值上升最快的方向 → 沿负梯度就走到最小值。\n")

# 学习率对比：走大步还是走小步？
print("--- 学习率权衡: 走大步还是走小步？ ---")
print(f"同一个 f(x)=3x²-4x (谷底 x=0.667)，从 x=5 出发:\n")

def demo_lr(lr, steps=15):
    """用指定学习率从 x=5 出发，追踪路径"""
    x = torch.tensor(5.0, requires_grad=True)
    path = [x.item()]
    for _ in range(steps):
        y = 3 * x**2 - 4 * x
        y.backward()
        with torch.no_grad():
            x -= lr * x.grad
        x.grad.zero_()
        path.append(x.item())
        if abs(x.item() - 0.667) < 0.001:
            break
    return path

for lr_val, desc in [(0.001, '太小: 龟速挪动'), (0.1, '适中: 自然收敛'),
                      (0.25, '偏大: 左右震荡'), (1.2, '太大: 永不收敛')]:
    path = demo_lr(lr_val)
    first5 = [f"{p:+.1f}" for p in path[:6]]
    print(f"  lr={lr_val:<6} ({desc})")
    print(f"    {len(path)}步 → x={path[-1]:.4f}  路径: {first5}{'...' if len(path)>6 else ''}")

print(f"\n直观对比:")
print(f"  lr=0.001: ●→●→●→●  每步挪 0.03，15步都不见影")
print(f"  lr=0.1:   ●→●→●→●  10步精准到达谷底 ✓")
print(f"  lr=0.25:  ●↗●↙●↗●  跨过谷底飞回来，震荡")
print(f"  lr=1.2:   ●↗↗↗↗  一步跨过谷底飞向无穷 ✗")
print(f"\n规律:")
print(f"  坡度大(高处) → 步幅大 → 快速下山")
print(f"  坡度小(近谷底) → 步幅小 → 自然减速不冲过头")
print(f"  学习率太大 → 步幅超过'谷底宽度' → 永远跨不过去")
print(f"\n实践: lr=0.1 开局, 每30轮衰减10倍。大步快走 + 小步精调。\n")


# ====== 2.4.4 链式法则 ======
print(f"\n===== 2.4.4 链式法则 =====\n")

# 练习4: u = f(x, y, z), 其中 x=x(a,b), y=y(a,b), z=z(a,b)
print("练习4: 多元链式法则")
print("u = f(x(a,b), y(a,b), z(a,b))")
print()
print("∂u/∂a = ∂u/∂x · ∂x/∂a + ∂u/∂y · ∂y/∂a + ∂u/∂z · ∂z/∂a")
print("∂u/∂b = ∂u/∂x · ∂x/∂b + ∂u/∂y · ∂y/∂b + ∂u/∂z · ∂z/∂b")
print()
print("规律: 对参数求导时，把所有从该参数通向输出的路径梯度相加。")

# 具体示例验证
def verify_chain_rule():
    """用 PyTorch 验证多路径链式法则"""
    a = torch.tensor(1.0, requires_grad=True)
    b = torch.tensor(2.0, requires_grad=True)

    x = a**2 + b           # x = a² + b
    y = a * b              # y = a·b
    z = torch.exp(a + b)   # z = e^{a+b}
    u = x + 2*y + z        # u = x + 2y + z

    u.backward()

    print(f"\n具体验证: u = (a²+b) + 2(a·b) + e^(a+b)")
    print(f"du/da (PyTorch) = {a.grad:.4f}")
    # du/da = dx/da + 2·dy/da + dz/da
    #        = 2a + 2·b + e^(a+b)·1
    #        = 2 + 4 + e³ = 6 + 20.0855 = 26.0855
    theory_a = 2*1 + 2*2 + np.exp(3)
    print(f"du/da (理论)    = 2a + 2b + e^(a+b) = {theory_a:.4f}")
    print(f"一致: {abs(a.grad.item() - theory_a) < 1e-4}")

verify_chain_rule()


# ====== 2.5.1 自动求导 ======
print(f"\n===== 2.5.1 自动求导 =====\n")

# requires_grad: 告诉 PyTorch 跟踪这个张量的操作
x = torch.arange(4.0, requires_grad=True)
print(f"x = {x}, requires_grad = {x.requires_grad}")

# y = 2·xᵀx = 2·||x||²
y = 2 * torch.dot(x, x)
print(f"y = 2·xᵀx = 2·||x||² = {y.item():.0f}")

y.backward()
print(f"\nx.grad = {x.grad}")
print(f"理论: ∂y/∂x = 4x = {4 * x}")
print(f"匹配: {torch.allclose(x.grad, 4 * x)}")


# ====== 2.5.2 梯度累积与清零 ======
print(f"\n===== 2.5.2 梯度累积 =====\n")

# backward 默认累加梯度——训练前必须清零
x = torch.tensor([1.0, 2.0], requires_grad=True)

y1 = x.sum()
y1.backward()
print(f"第1次 backward (y₁=x.sum()): grad = {x.grad}")

# 不清零直接再来一次
y2 = (x ** 2).sum()
y2.backward()
print(f"第2次 backward (y₂=x².sum()) 不清零: grad = {x.grad}")
print("→ 梯度被累加了！训练循环中必须 optimizer.zero_grad() 或 x.grad.zero_()")

x.grad.zero_()
y3 = x.sum()
y3.backward()
print(f"清零后 backward: grad = {x.grad}")


# ====== 2.5.3 非标量反向传播 ======
print(f"\n===== 2.5.3 非标量反向传播 =====\n")

x = torch.arange(4.0, requires_grad=True)
y = x * x

# .backward() 只能对标量调用，y 是向量需要先转标量
print(f"y = x*x = {y.tolist()} ← 向量，不能直接 .backward()")
y.sum().backward()
print(f"y.sum().backward() → grad = {x.grad}")
print(f"理论 2x = {2 * x.detach()}")


# ====== 2.5.4 分离计算 (detach) ======
print(f"\n===== 2.5.4 分离计算 =====\n")

x = torch.arange(4.0, requires_grad=True)
y = x * x
u = y.detach()    # u 脱离计算图，视为常数
z = u * x         # z = (常数 x²) * x → ∂z/∂x = u = x²

z.sum().backward()
print(f"z = detach(x²) * x")
print(f"grad = {x.grad}, 理论 x² = {x**2}")
print(f"匹配: {torch.allclose(x.grad, x**2)}")


# ====== 2.5.5 Python 控制流 ======
print(f"\n===== 2.5.5 Python 控制流 =====\n")

# PyTorch 动态计算图: Python 控制流也可以求导
def f_control(a):
    """带 while 和 if 的函数，仍可自动求导"""
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c

# 标量版本
a = torch.randn(size=(), requires_grad=True)
d = f_control(a)
d.backward()

# 数值验证
h = 1e-4
a_val = a.detach().item()
with torch.no_grad():
    f_plus = f_control(torch.tensor(a_val + h))
    f_minus = f_control(torch.tensor(a_val - h))
    num_grad = (f_plus - f_minus) / (2 * h)

print(f"a = {a.item():.4f}, d = f(a) = {d.item():.4f}")
print(f"自动求导 grad = {a.grad.item():.4f}")
print(f"数值验证 grad = {num_grad:.4f}")
rel_err = abs(a.grad.item() - num_grad) / abs(a.grad.item())
print(f"相对误差 = {rel_err:.6f} ({rel_err*100:.4f}%) — 一致!")


# ====== 练习 (2.4 微积分) ======
print(f"\n===== 练习 (2.4 微积分) =====\n")

# 练习1: 绘制 f(x) = x³ - 1/x 及其在 x=1 处的切线
print("--- 练习1: f(x) = x³ - 1/x 及其 x=1 处切线 ---")

def f_ex1(x):
    return x ** 3 - 1 / x

# f'(x) = 3x² + 1/x²
# 在 x=1: f(1) = 0, f'(1) = 4
# 切线: y = 4(x-1) + 0 = 4x - 4

print("f(x) = x³ - 1/x")
print("f'(x) = 3x² + 1/x²")
print("在 x=1: f(1) = 0, f'(1) = 4")
print("切线: y = 4x - 4\n")

# 注意: x > 0 避免除以零
x = np.arange(0.1, 3, 0.1)
fig, ax = plt.subplots(figsize=(6, 4))
plot(x, [f_ex1(x), 4 * x - 4], 'x', 'f(x)',
     legend=[r'$f(x) = x^3 - 1/x$', 'Tangent line (x=1): y = 4x - 4'],
     axes=ax)
ax.annotate('(1, 0)', xy=(1, 0), xytext=(1.5, 1),
            arrowprops=dict(arrowstyle='->'), fontsize=10)

ex1_path = os.path.join(data_dir, 'ex1_x3_minus_1overx.png')
plt.savefig(ex1_path, dpi=120)
print(f"图像已保存: {ex1_path}")

# 练习2: 求 f(x) = 3x1^2 + 5e^{x2} 的梯度 (已在上方 2.4.2 节计算)
print(f"\n--- 练习2: f(x) = 3x1^2 + 5·exp(x2) 的梯度 ---")
x = torch.tensor([1.0, 2.0], requires_grad=True)
f_val = 3 * x[0]**2 + 5 * torch.exp(x[1])
f_val.backward()
print(f"梯度 ∇f = [df/dx1, df/dx2] = [{x.grad[0]:.4f}, {x.grad[1]:.4f}]")
print("理论: df/dx1 = 6*x1, df/dx2 = 5*exp(x2)")

# 练习3: f(x) = ||x||₂ 的梯度 (已在上方 2.4.3 节计算)
print(f"\n--- 练习3: f(x) = ||x||₂ 的梯度 (已在上方计算) ---")
print("结果: ∇||x||₂ = x / ||x||₂, 即单位向量方向")

# 练习4: 链式法则 (已在上方 2.4.4 节计算)
print(f"\n--- 练习4: 多元链式法则 (已在上方验证) ---")
print("∂u/∂a = Σ ∂u/∂_ · ∂_/∂a   (所有从 a 出发路径的梯度求和)")


# ====== 练习 (2.5 自动微分) ======
print(f"\n===== 练习 (2.5 自动微分) =====\n")

# 练习1: 为什么计算二阶导数比一阶导数开销更大？
print("--- 练习1: 二阶导数的开销 ---")

x = torch.arange(4.0, requires_grad=True)
y = 2 * torch.dot(x, x)

# 一阶导数
grad1 = torch.autograd.grad(y, x, create_graph=True)[0]
print(f"一阶导数: {grad1}")

# 二阶导数: 需要对一阶导数再求导
grad2 = torch.autograd.grad(grad1.sum(), x)[0]
print(f"二阶导数: {grad2} (应全为 4，因 y=2Σx² 是二次函数)")

print("\n存储开销:")
print(f"  y 占内存: {y.element_size() * y.nelement()} bytes")
print(f"  一阶导占内存: {grad1.element_size() * grad1.nelement()} bytes")
print(f"  总 intermediate 大小随 create_graph=True 翻倍")
print("结论: 二阶导需要 create_graph=True 保留计算图，内存翻倍，计算翻倍。")
print("在深度学习中百万参数的二阶导(Hessian)完全不可行。")

# 练习2: 立即再次运行 backward 会发生什么？
print(f"\n--- 练习2: 连续两次 backward ---")

x = torch.arange(4.0, requires_grad=True)
y = 2 * torch.dot(x, x)
y.backward()
print(f"第1次 backward: grad = {x.grad}")

try:
    y.backward()
except RuntimeError as e:
    print(f"第2次 backward 报错: {str(e)[:60]}...")
    print("→ 计算图已释放！需要 retain_graph=True 或重新前向传播")

# 正确做法: 重新构建
y = 2 * torch.dot(x, x)
y.backward()
print(f"重建图后 backward: grad = {x.grad} (累加了)")

# 练习3: 控制流中 a 改为向量或矩阵
print(f"\n--- 练习3: 控制流对向量/矩阵求导 ---")

# 向量版本
a_vec = torch.randn(4, requires_grad=True)
d_vec = f_control(a_vec)
d_vec.sum().backward()  # 向量需要先转标量
print(f"a_vec = {a_vec}")
print(f"d_vec = f(a_vec)")
print(f"grad = {a_vec.grad}")
print("结论: 控制流对向量/矩阵同样有效，PyTorch 每元素独立跟踪计算图。")

# 练习4: 重新设计一个控制流求导例子
print(f"\n--- 练习4: 自定义控制流求导 ---")

def f_custom(a):
    """一个更复杂的控制流: 分段函数 + 循环"""
    # 分段函数: a <= 0 走一条路, a > 0 走另一条
    if a.sum() > 0:
        # 正数路径: a 反复平方直到超过 10
        b = a
        while b.max() < 10:
            b = b * 1.5
        c = b.sum()
    else:
        # 负数/零路径: 用 tanh 压缩
        c = torch.tanh(a).sum() * 10
    return c

a_test = torch.randn(3, requires_grad=True)
d_test = f_custom(a_test)
d_test.backward()

# 数值验证: 逐元素计算
h = 1e-4
num_grad = []
with torch.no_grad():
    for i in range(len(a_test)):
        a_plus = a_test.clone()
        a_minus = a_test.clone()
        a_plus[i] += h
        a_minus[i] -= h
        d_plus = f_custom(a_plus)
        d_minus = f_custom(a_minus)
        num_grad.append(((d_plus - d_minus) / (2 * h)).item())

print(f"a = {a_test.tolist()}")
print(f"d = f(a) = {d_test.item():.4f}")
print(f"自动求导 grad = {a_test.grad.tolist()}")
print(f"数值验证 grad = {num_grad}")
max_rel_err = max(abs(g - n) / max(abs(g), 1) for g, n in zip(a_test.grad.tolist(), num_grad))
print(f"最大相对误差 = {max_rel_err:.6f} — {'一致 ✓' if max_rel_err < 0.01 else '需检查'}")

# 练习5: 用自动求导绘制 f(x)=sin(x) 和 df/dx (不用 cos(x)!)
print(f"\n--- 练习5: 用 autograd 绘制 sin(x) 及其导数 (不用 cos 公式) ---")
print("d2l 原题: 绘制 f(x)=sin(x) 和 df/dx，后者不使用 f'(x)=cos(x)")
print("关键: 用 PyTorch 的 autograd 来数值计算导数，而不是套 cos 公式\n")

# 在 x 轴上取 200 个点，每个点用 autograd 算出导数
x_vals = np.linspace(-2 * np.pi, 2 * np.pi, 200)
sin_vals = np.sin(x_vals)

# 用 autograd 逐点计算导数: 对每个 x_i，算 f(x_i) 然后 backward 得到 f'(x_i)
autograd_derivs = []
for xi in x_vals:
    x_t = torch.tensor(xi, requires_grad=True)  # 每个点独立建图
    y_t = torch.sin(x_t)                        # f(x) = sin(x)
    y_t.backward()                              # autograd 自动算导数
    autograd_derivs.append(x_t.grad.item())     # 取出梯度 = f'(x_i)

# 对比: 解析公式 cos(x)（仅用于验证 autograd 对不对）
cos_vals = np.cos(x_vals)

# 绘图
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# 上图: f(x)=sin(x)
ax1.plot(x_vals, sin_vals, 'b-', linewidth=2, label=r'$f(x) = \sin(x)$')
ax1.set_ylabel(r'$f(x)$', fontsize=12)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_title(r'Exercise 5: $\sin(x)$ and its derivative (computed by autograd, NOT $\cos(x)$ formula)', fontsize=13)

# 下图: autograd 算出的导数 vs cos(x) 理论值
ax2.plot(x_vals, autograd_derivs, 'r-', linewidth=2.5, alpha=0.7,
         label='Autograd derivative (numerical)')
ax2.plot(x_vals, cos_vals, 'b--', linewidth=1.5, alpha=0.8,
         label=r'$\cos(x)$ (analytical, for verification only)')
ax2.set_xlabel('x', fontsize=12)
ax2.set_ylabel(r"$f'(x)$", fontsize=12)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

# 验证: autograd 和 cos 的最大误差
errors = np.abs(np.array(autograd_derivs) - cos_vals)
ax2.fill_between(x_vals, cos_vals - errors, cos_vals + errors,
                  alpha=0.15, color='gray', label=f'Max error: {errors.max():.2e}')
ax2.legend(fontsize=10)

plt.tight_layout()
ex5_path = os.path.join(data_dir, 'ex5_sin_autograd.png')
plt.savefig(ex5_path, dpi=120)
print(f"图像已保存: {ex5_path}")
print(f"\n验证: autograd 数值导数 vs cos(x) 解析公式")
print(f"  最大误差 = {errors.max():.2e}")
print(f"  平均误差 = {errors.mean():.2e}")
print("结论: autograd 算出的导数与 cos(x) 几乎完全一致（误差来自 float 精度）。")
print("这证明了自动求导的正确性——你不用推导 cos(x) 公式，PyTorch 直接给你精确的导数。")


print(f"\n===== 完成 =====")
print("微积分 + 自动求导总结:")
print("  导数 → 偏导数 → 梯度 → 链式法则 → 自动求导 → 梯度下降")
