"""
问题2：三次样条插值 - 船舶吃水深度曲线

作者：数值计算课程
日期：2025年10月
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False


def solve_tridiagonal(a, b, c, d):
    """
    追赶法求解三对角方程组
    
    参数:
        a: 下对角线系数（长度为n-1）
        b: 主对角线系数（长度为n）
        c: 上对角线系数（长度为n-1）
        d: 右端项（长度为n）
    
    返回:
        x: 解向量
    """
    n = len(b)
    c_new = np.zeros(n-1)
    d_new = np.zeros(n)
    x = np.zeros(n)
    
    # 前向消元
    c_new[0] = c[0] / b[0]
    d_new[0] = d[0] / b[0]
    
    for i in range(1, n-1):
        temp = b[i] - a[i-1] * c_new[i-1]
        c_new[i] = c[i] / temp
        d_new[i] = (d[i] - a[i-1] * d_new[i-1]) / temp
    
    d_new[n-1] = (d[n-1] - a[n-2] * d_new[n-2]) / (b[n-1] - a[n-2] * c_new[n-2])
    
    # 回代
    x[n-1] = d_new[n-1]
    for i in range(n-2, -1, -1):
        x[i] = d_new[i] - c_new[i] * x[i+1]
    
    return x


def cubic_spline_coefficients(x, y):
    """
    计算三次样条插值的系数
    
    参数:
        x: 节点位置数组（长度为n+1）
        y: 节点值数组（长度为n+1）
    
    返回:
        M: 各节点的二阶导数值
        h: 各区间的长度
    """
    n = len(x) - 1
    h = np.diff(x)  # 各区间长度
    
    # 构造三对角方程组
    # 使用自然边界条件: M[0] = M[n] = 0
    
    if n == 1:
        # 只有两个点，直接返回
        return np.array([0.0, 0.0]), h
    
    # 内部节点方程 (n-1个方程)
    a = np.zeros(n-2)  # 下对角线 (n-2个元素)
    b = np.zeros(n-1)  # 主对角线 (n-1个元素)
    c = np.zeros(n-2)  # 上对角线 (n-2个元素)
    d = np.zeros(n-1)  # 右端项 (n-1个元素)
    
    # 内部节点方程
    for i in range(n-1):
        if i > 0:
            a[i-1] = h[i]
        b[i] = 2 * (h[i] + h[i+1])
        if i < n-2:
            c[i] = h[i+1]
        d[i] = 6 * ((y[i+2] - y[i+1]) / h[i+1] - (y[i+1] - y[i]) / h[i])
    
    # 求解三对角方程组得到内部节点的M值
    M_inner = solve_tridiagonal(a, b, c, d)
    
    # 加上边界条件
    M = np.zeros(n+1)
    M[0] = 0.0  # 自然边界条件
    M[1:n] = M_inner
    M[n] = 0.0  # 自然边界条件
    
    return M, h


def cubic_spline_interpolate(x, y, x_new):
    """
    三次样条插值
    
    参数:
        x: 原始节点位置
        y: 原始节点值
        x_new: 需要插值的新位置
    
    返回:
        y_new: 插值结果
    """
    M, h = cubic_spline_coefficients(x, y)
    y_new = np.zeros_like(x_new)
    
    for k, xk in enumerate(x_new):
        # 找到xk所在的区间
        if xk <= x[0]:
            i = 0
        elif xk >= x[-1]:
            i = len(x) - 2
        else:
            i = np.searchsorted(x, xk) - 1
        
        # 确保i在有效范围内
        i = max(0, min(i, len(x) - 2))
        
        # 计算样条值
        dx = xk - x[i]
        dx_end = x[i+1] - xk
        
        y_new[k] = (M[i] * dx_end**3 / (6 * h[i]) + 
                    M[i+1] * dx**3 / (6 * h[i]) +
                    (y[i] - M[i] * h[i]**2 / 6) * dx_end / h[i] +
                    (y[i+1] - M[i+1] * h[i]**2 / 6) * dx / h[i])
    
    return y_new


def plot_results(x, y, x_new, y_new):
    """
    绘制原始数据点和插值曲线
    
    参数:
        x: 原始节点位置
        y: 原始节点值
        x_new: 插值位置
        y_new: 插值结果
    """
    plt.figure(figsize=(12, 7))
    
    # 绘制插值曲线
    plt.plot(x_new, y_new, 'b-', linewidth=2, label='三次样条插值曲线')
    
    # 绘制原始数据点
    plt.plot(x, y, 'ro', markersize=10, label='测量点', zorder=5)
    
    # 标注测量点
    for i, (xi, yi) in enumerate(zip(x, y)):
        plt.annotate(f'({xi}, {yi})', 
                     xy=(xi, yi), 
                     xytext=(5, 5),
                     textcoords='offset points',
                     fontsize=9)
    
    plt.xlabel('船长方向位置 (米)', fontsize=12)
    plt.ylabel('吃水深度 (米)', fontsize=12)
    plt.title('船舶吃水深度三次样条插值', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('spline_interpolation.png', dpi=300, bbox_inches='tight')
    print("插值曲线图已保存为 'spline_interpolation.png'")
    plt.close()


def calculate_errors(x, y):
    """
    计算插值误差（在原始节点处）
    
    参数:
        x: 原始节点位置
        y: 原始节点值
    
    返回:
        errors: 各点的插值误差
    """
    y_interp = cubic_spline_interpolate(x, y, x)
    errors = np.abs(y - y_interp)
    return errors


def local_adjustment_demo(x, y):
    """
    演示局部调整功能
    
    参数:
        x: 原始节点位置
        y: 原始节点值
    """
    print("\n" + "="*60)
    print("局部调整演示")
    print("="*60)
    
    # 原始插值
    x_fine = np.linspace(x[0], x[-1], 500)
    y_original = cubic_spline_interpolate(x, y, x_fine)
    
    # 在区间[40, 60]增加0.2米
    y_adjusted = y.copy()
    for i in range(len(x)):
        if 40 <= x[i] <= 60:
            y_adjusted[i] += 0.2
            print(f"调整节点 {i}: x={x[i]}, y: {y[i]} -> {y_adjusted[i]}")
    
    # 调整后的插值
    y_new = cubic_spline_interpolate(x, y_adjusted, x_fine)
    
    # 绘制对比图
    plt.figure(figsize=(12, 7))
    plt.plot(x_fine, y_original, 'b-', linewidth=2, label='原始曲线')
    plt.plot(x_fine, y_new, 'r--', linewidth=2, label='调整后曲线')
    plt.plot(x, y, 'bo', markersize=10, label='原始测量点')
    plt.plot(x, y_adjusted, 'rs', markersize=10, label='调整后测量点')
    
    # 标注调整区域
    plt.axvspan(40, 60, alpha=0.2, color='yellow', label='调整区域')
    
    plt.xlabel('船长方向位置 (米)', fontsize=12)
    plt.ylabel('吃水深度 (米)', fontsize=12)
    plt.title('局部调整对比图', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('local_adjustment.png', dpi=300, bbox_inches='tight')
    print("\n局部调整对比图已保存为 'local_adjustment.png'")
    plt.close()
    
    # 分析影响范围
    diff = np.abs(y_new - y_original)
    max_diff_idx = np.argmax(diff)
    print(f"\n最大变化位置: x={x_fine[max_diff_idx]:.2f}米, 变化量={diff[max_diff_idx]:.4f}米")
    print(f"在调整区域内的平均变化: {np.mean(diff[(x_fine >= 40) & (x_fine <= 60)]):.4f}米")
    print(f"在调整区域外的平均变化: {np.mean(diff[(x_fine < 40) | (x_fine > 60)]):.4f}米")


def main():
    """
    主函数
    """
    print("="*60)
    print("问题2：船舶吃水深度三次样条插值")
    print("="*60)
    
    # 测量数据
    x = np.array([0, 20, 40, 60, 80, 100], dtype=float)  # 位置（米）
    y = np.array([5.0, 5.5, 6.2, 5.8, 5.3, 5.0], dtype=float)  # 吃水深度（米）
    
    print(f"\n测量点数据:")
    print(f"位置 (米): {x}")
    print(f"吃水深度 (米): {y}")
    
    # 计算样条系数
    M, h = cubic_spline_coefficients(x, y)
    
    print(f"\n三次样条系数计算结果:")
    print(f"各节点二阶导数 M:")
    for i, mi in enumerate(M):
        print(f"  M[{i}] = {mi:.6f}")
    
    print(f"\n各区间长度 h:")
    for i, hi in enumerate(h):
        print(f"  h[{i}] = {hi:.2f}")
    
    # 生成密集插值点
    x_new = np.linspace(x[0], x[-1], 500)
    y_new = cubic_spline_interpolate(x, y, x_new)
    
    # 绘制结果
    plot_results(x, y, x_new, y_new)
    
    # 计算插值误差
    errors = calculate_errors(x, y)
    print(f"\n插值误差分析（在原始节点处）:")
    print(f"最大误差: {np.max(errors):.2e}")
    print(f"平均误差: {np.mean(errors):.2e}")
    print(f"说明: 理论上在节点处误差为0，实际误差为数值计算误差")
    
    # 计算并显示一些特定位置的插值结果
    print(f"\n特定位置的插值结果:")
    test_points = [10, 30, 50, 70, 90]
    for xp in test_points:
        yp = cubic_spline_interpolate(x, y, np.array([xp]))[0]
        print(f"  x = {xp:3d} 米 -> y = {yp:.4f} 米")
    
    # 局部调整演示
    local_adjustment_demo(x, y)
    
    print("\n" + "="*60)
    print("计算完成！")
    print("="*60)


if __name__ == "__main__":
    main()
