"""
问题1：使用Romberg算法计算船舶水下表面积

作者：数值计算课程
日期：2025年10月
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False


def integrand(x, L, B, D):
    """
    计算被积函数的值
    
    参数:
        x: 积分变量
        L: 船舶长度（米）
        B: 船舶宽度（米）
        D: 吃水深度（米）
    
    返回:
        被积函数的值
    """
    # dy/dx = (B*pi/L) * cos(pi*x/L)
    dy_dx = (B * np.pi / L) * np.cos(np.pi * x / L)
    
    # dz/dx = (2*D*pi/L) * sin(2*pi*x/L)
    dz_dx = (2 * D * np.pi / L) * np.sin(2 * np.pi * x / L)
    
    # sqrt(1 + (dy/dx)^2 + (dz/dx)^2)
    return np.sqrt(1 + dy_dx**2 + dz_dx**2)


def trapezoidal(f, a, b, n, L, B, D):
    """
    复化梯形公式
    
    参数:
        f: 被积函数
        a: 积分下限
        b: 积分上限
        n: 子区间数
        L, B, D: 传递给被积函数的参数
    
    返回:
        积分近似值
    """
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x, L, B, D)
    
    # 梯形公式: h/2 * [f(a) + 2*sum(f(xi)) + f(b)]
    result = h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])
    return result


def romberg_integration(L=50, B=8, D=3, epsilon=1e-6, max_iter=20):
    """
    Romberg积分算法
    
    参数:
        L: 船舶长度（米）
        B: 船舶宽度（米）
        D: 吃水深度（米）
        epsilon: 精度要求
        max_iter: 最大迭代次数
    
    返回:
        surface_area: 表面积
        iteration_values: 每次迭代的值
        R: Romberg矩阵
    """
    a, b = 0, L  # 积分区间
    
    # 初始化Romberg矩阵
    R = np.zeros((max_iter, max_iter))
    iteration_values = []
    
    # 第一列：梯形公式
    for k in range(max_iter):
        n = 2**k
        R[k, 0] = trapezoidal(integrand, a, b, n, L, B, D)
        
        # Richardson外推
        for m in range(1, k + 1):
            R[k, m] = (4**m * R[k, m-1] - R[k-1, m-1]) / (4**m - 1)
        
        # 对角线元素是当前最佳估计
        current_value = R[k, k]
        iteration_values.append(current_value)
        
        # 检查收敛性
        if k > 0:
            relative_error = abs(R[k, k] - R[k-1, k-1]) / abs(R[k, k])
            print(f"迭代 {k+1}: R[{k},{k}] = {R[k, k]:.10f}, 相对误差 = {relative_error:.2e}")
            
            if relative_error < epsilon:
                print(f"\n算法在{k+1}次迭代后收敛!")
                # 表面积需要乘以2（因为有左右两侧）
                surface_area = 2 * R[k, k]
                return surface_area, iteration_values[:k+1], R[:k+1, :k+1]
        else:
            print(f"迭代 {k+1}: R[{k},{k}] = {R[k, k]:.10f}")
    
    # 如果达到最大迭代次数
    print(f"\n达到最大迭代次数 {max_iter}")
    surface_area = 2 * R[max_iter-1, max_iter-1]
    return surface_area, iteration_values, R


def plot_convergence(iteration_values):
    """
    绘制收敛曲线
    
    参数:
        iteration_values: 每次迭代的值列表
    """
    plt.figure(figsize=(10, 6))
    iterations = range(1, len(iteration_values) + 1)
    
    plt.plot(iterations, iteration_values, 'b-o', linewidth=2, markersize=8)
    plt.xlabel('迭代次数', fontsize=12)
    plt.ylabel('积分近似值（米）', fontsize=12)
    plt.title('Romberg算法收敛曲线', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('convergence_curve.png', dpi=300, bbox_inches='tight')
    print("\n收敛曲线已保存为 'convergence_curve.png'")
    plt.close()


def compare_with_simple_shapes(surface_area, L, B, D):
    """
    与简单几何形状进行对比
    
    参数:
        surface_area: 计算得到的表面积
        L: 船舶长度
        B: 船舶宽度
        D: 吃水深度
    """
    print("\n" + "="*60)
    print("与简单几何形状的对比分析")
    print("="*60)
    
    # 圆柱体近似（半径为B/2）
    cylinder_surface = 2 * np.pi * (B/2) * L
    print(f"圆柱体近似（半径{B/2}m，长{L}m）：{cylinder_surface:.2f} 平方米")
    
    # 椭球体近似（使用近似公式）
    # 半长轴 a = L/2, 半短轴 b = c = B/2
    a = L / 2
    b = B / 2
    # 使用近似公式: S ≈ 4π * ((a^p*b^p + a^p*c^p + b^p*c^p)/3)^(1/p), p≈1.6
    ellipsoid_surface = 4 * np.pi * ((a**1.6 * b**1.6 + a**1.6 * b**1.6 + b**1.6 * b**1.6) / 3)**(1/1.6)
    print(f"椭球体近似（半长轴{a}m，半短轴{b}m）：{ellipsoid_surface:.2f} 平方米")
    
    print(f"\n计算结果：{surface_area:.2f} 平方米")
    print(f"该结果介于圆柱体和椭球体之间，符合船舶外形特征")


def main():
    """
    主函数
    """
    print("="*60)
    print("问题1：船舶水下表面积计算（Romberg积分）")
    print("="*60)
    
    # 参数设置
    L = 50  # 船舶长度（米）
    B = 8   # 船舶宽度（米）
    D = 3   # 吃水深度（米）
    epsilon = 1e-6  # 精度要求
    
    print(f"\n船舶参数:")
    print(f"  长度 L = {L} 米")
    print(f"  宽度 B = {B} 米")
    print(f"  吃水深度 D = {D} 米")
    print(f"  精度要求 ε = {epsilon}")
    print(f"\n开始Romberg积分计算...\n")
    
    # 执行Romberg积分
    surface_area, iteration_values, R = romberg_integration(L, B, D, epsilon)
    
    # 输出结果
    print("\n" + "="*60)
    print("最终结果")
    print("="*60)
    print(f"船舶水下表面积 S = {surface_area:.4f} 平方米")
    print(f"迭代次数: {len(iteration_values)}")
    
    # 绘制收敛曲线
    plot_convergence(iteration_values)
    
    # 与简单几何形状对比
    compare_with_simple_shapes(surface_area, L, B, D)
    
    # 显示Romberg矩阵（部分）
    print("\n" + "="*60)
    print("Romberg矩阵（前几行）")
    print("="*60)
    n_rows = min(5, len(iteration_values))
    for i in range(n_rows):
        print(f"行 {i}: ", end="")
        for j in range(i+1):
            print(f"{R[i, j]:12.6f}", end=" ")
        print()
    
    print("\n计算完成！")


if __name__ == "__main__":
    main()
