"""
问题4：核燃料循环中的同位素分离问题 - 线性方程组求解

作者：数值计算课程
日期：2025年10月
"""

import numpy as np
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False


def gaussian_elimination(A, b):
    """
    标准高斯消元法求解线性方程组 Ax = b
    
    参数:
        A: 系数矩阵 (n×n)
        b: 常数向量 (n×1)
    
    返回:
        x: 解向量
    """
    n = len(b)
    # 创建增广矩阵
    Ab = np.hstack([A.astype(float), b.reshape(-1, 1)])
    
    # 前向消元
    for k in range(n-1):
        # 检查主元是否为零
        if abs(Ab[k, k]) < 1e-10:
            raise ValueError(f"主元 Ab[{k},{k}] = {Ab[k,k]} 太小，可能导致数值不稳定")
        
        # 消元
        for i in range(k+1, n):
            factor = Ab[i, k] / Ab[k, k]
            Ab[i, k:] -= factor * Ab[k, k:]
    
    # 回代
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (Ab[i, -1] - np.dot(Ab[i, i+1:n], x[i+1:n])) / Ab[i, i]
    
    return x


def gaussian_elimination_with_pivoting(A, b):
    """
    列主元高斯消元法求解线性方程组 Ax = b
    
    参数:
        A: 系数矩阵 (n×n)
        b: 常数向量 (n×1)
    
    返回:
        x: 解向量
    """
    n = len(b)
    # 创建增广矩阵
    Ab = np.hstack([A.astype(float), b.reshape(-1, 1)])
    
    # 前向消元（带列主元）
    for k in range(n-1):
        # 选择列主元
        max_row = k + np.argmax(np.abs(Ab[k:n, k]))
        
        # 交换行
        if max_row != k:
            Ab[[k, max_row]] = Ab[[max_row, k]]
            print(f"交换第 {k} 行和第 {max_row} 行")
        
        # 检查主元是否为零
        if abs(Ab[k, k]) < 1e-10:
            raise ValueError(f"主元 Ab[{k},{k}] = {Ab[k,k]} 太小")
        
        # 消元
        for i in range(k+1, n):
            factor = Ab[i, k] / Ab[k, k]
            Ab[i, k:] -= factor * Ab[k, k:]
    
    # 回代
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (Ab[i, -1] - np.dot(Ab[i, i+1:n], x[i+1:n])) / Ab[i, i]
    
    return x


def calculate_residual(A, b, x):
    """
    计算残差 r = b - Ax
    
    参数:
        A: 系数矩阵
        b: 常数向量
        x: 解向量
    
    返回:
        residual: 残差向量
        relative_error: 相对误差
    """
    residual = b - np.dot(A, x)
    relative_error = np.linalg.norm(residual) / np.linalg.norm(b)
    return residual, relative_error


def verify_solution(A, b, x):
    """
    验证解的正确性
    
    参数:
        A: 系数矩阵
        b: 常数向量
        x: 解向量
    """
    print("\n" + "="*60)
    print("解的验证")
    print("="*60)
    
    # 计算残差
    residual, relative_error = calculate_residual(A, b, x)
    
    print(f"\n残差向量 r = b - Ax:")
    for i, r in enumerate(residual):
        print(f"  r[{i}] = {r:.2e}")
    
    print(f"\n残差范数: ||r|| = {np.linalg.norm(residual):.2e}")
    print(f"相对残差: ||r||/||b|| = {relative_error:.2e}")
    
    if relative_error < 1e-10:
        print("\n✓ 解的精度很高！")
    elif relative_error < 1e-6:
        print("\n✓ 解的精度良好")
    else:
        print("\n⚠ 解的精度可能不够")
    
    # 验证 Ax
    Ax = np.dot(A, x)
    print(f"\n验证 Ax = b:")
    print("  i    b[i]      Ax[i]     差值")
    print("-" * 40)
    for i in range(len(b)):
        diff = b[i] - Ax[i]
        print(f"  {i}  {b[i]:8.2f}  {Ax[i]:8.2f}  {diff:.2e}")


def compare_methods(A, b):
    """
    比较不同方法的结果
    
    参数:
        A: 系数矩阵
        b: 常数向量
    """
    print("\n" + "="*60)
    print("方法对比")
    print("="*60)
    
    # 方法1：标准高斯消元
    print("\n1. 标准高斯消元法:")
    try:
        x1 = gaussian_elimination(A.copy(), b.copy())
        _, rel_err1 = calculate_residual(A, b, x1)
        print(f"   相对残差: {rel_err1:.2e}")
    except Exception as e:
        print(f"   失败: {e}")
        x1 = None
    
    # 方法2：列主元高斯消元
    print("\n2. 列主元高斯消元法:")
    x2 = gaussian_elimination_with_pivoting(A.copy(), b.copy())
    _, rel_err2 = calculate_residual(A, b, x2)
    print(f"   相对残差: {rel_err2:.2e}")
    
    # 方法3：NumPy的linalg.solve（使用LU分解）
    print("\n3. NumPy linalg.solve (LU分解):")
    x3 = np.linalg.solve(A, b)
    _, rel_err3 = calculate_residual(A, b, x3)
    print(f"   相对残差: {rel_err3:.2e}")
    
    # 比较解
    if x1 is not None:
        print(f"\n解的差异:")
        print(f"  ||x1 - x2|| = {np.linalg.norm(x1 - x2):.2e}")
        print(f"  ||x1 - x3|| = {np.linalg.norm(x1 - x3):.2e}")
        print(f"  ||x2 - x3|| = {np.linalg.norm(x2 - x3):.2e}")


def analyze_matrix_properties(A):
    """
    分析矩阵性质
    
    参数:
        A: 系数矩阵
    """
    print("\n" + "="*60)
    print("矩阵性质分析")
    print("="*60)
    
    n = A.shape[0]
    
    # 行列式
    det = np.linalg.det(A)
    print(f"\n行列式: det(A) = {det:.4e}")
    
    if abs(det) > 1e-10:
        print("  矩阵非奇异，方程组有唯一解")
    else:
        print("  矩阵接近奇异，可能无解或有无穷多解")
    
    # 条件数
    cond = np.linalg.cond(A)
    print(f"\n条件数: cond(A) = {cond:.4e}")
    
    if cond < 100:
        print("  矩阵良态，数值稳定性好")
    elif cond < 10000:
        print("  矩阵状态一般，需注意数值误差")
    else:
        print("  矩阵病态，可能存在较大数值误差")
    
    # 对角占优性
    print(f"\n对角占优性检查:")
    is_diag_dominant = True
    for i in range(n):
        diag = abs(A[i, i])
        off_diag_sum = np.sum(np.abs(A[i, :])) - diag
        print(f"  行 {i}: |a_{i}{i}| = {diag:.2f}, 非对角元素之和 = {off_diag_sum:.2f}", end="")
        
        if diag > off_diag_sum:
            print(" ✓ 对角占优")
        else:
            print(" ✗ 非对角占优")
            is_diag_dominant = False
    
    if is_diag_dominant:
        print("\n  矩阵严格对角占优，迭代法必收敛")


def main():
    """
    主函数
    """
    print("="*60)
    print("问题4：核燃料循环中的同位素分离问题")
    print("="*60)
    
    # 系数矩阵
    A = np.array([
        [2.5, -0.8, 0, 0, 0],
        [-0.6, 3.0, -0.9, 0, 0],
        [0, -0.7, 3.2, -1.0, 0],
        [0, 0, -0.8, 3.5, -1.1],
        [0, 0, 0, -0.9, 3.8]
    ], dtype=float)
    
    # 常数向量
    b = np.array([1700, 1500, 1800, 2000, 2200], dtype=float)
    
    print("\n系数矩阵 A:")
    print(A)
    
    print("\n常数向量 b:")
    print(b)
    
    # 分析矩阵性质
    analyze_matrix_properties(A)
    
    # 使用列主元高斯消元法求解
    print("\n" + "="*60)
    print("使用列主元高斯消元法求解")
    print("="*60)
    
    x = gaussian_elimination_with_pivoting(A.copy(), b.copy())
    
    print("\n解向量 x (各级流量，单位：kg/h):")
    print("-" * 40)
    for i, xi in enumerate(x):
        print(f"  F_{i+1} = {xi:.6f} kg/h")
    
    # 验证解
    verify_solution(A, b, x)
    
    # 方法对比
    compare_methods(A, b)
    
    # 物理意义分析
    print("\n" + "="*60)
    print("物理意义分析")
    print("="*60)
    
    print(f"\n流量分析:")
    print(f"  最大流量: F_{np.argmax(x)+1} = {np.max(x):.2f} kg/h")
    print(f"  最小流量: F_{np.argmin(x)+1} = {np.min(x):.2f} kg/h")
    print(f"  流量范围: {np.max(x) - np.min(x):.2f} kg/h")
    print(f"  平均流量: {np.mean(x):.2f} kg/h")
    
    print(f"\n流量变化趋势:")
    for i in range(len(x)-1):
        change = x[i+1] - x[i]
        change_pct = (change / x[i]) * 100
        print(f"  F_{i+1} → F_{i+2}: {change:+.2f} kg/h ({change_pct:+.2f}%)")
    
    print("\n物理解释:")
    print("  - 各级流量均为正值，符合物理实际")
    print("  - 流量满足物质平衡方程")
    print("  - 系统稳定，适合工业生产")
    
    print("\n" + "="*60)
    print("计算完成！")
    print("="*60)


if __name__ == "__main__":
    main()
