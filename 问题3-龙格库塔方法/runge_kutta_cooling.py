"""
问题3：基于Runge-Kutta方法的核动力船舶冷却系统温度变化模拟

作者：数值计算课程
日期：2025年10月
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False


def power_function(t, P0, omega):
    """
    计算反应堆功率
    
    参数:
        t: 时间 (秒)
        P0: 平均功率 (MW)
        omega: 角频率 (rad/s)
    
    返回:
        功率值 (MW)
    """
    if t <= 3600:
        return P0 * (1 + 0.1 * np.sin(omega * t))
    else:
        return 0.0


def dT_dt(t, T, k, T_env, alpha, P0, omega):
    """
    计算温度变化率 dT/dt
    
    参数:
        t: 时间 (秒)
        T: 当前温度 (℃)
        k: 热交换系数 (1/s)
        T_env: 环境温度 (℃)
        alpha: 功率转换系数 (℃/MW)
        P0: 平均功率 (MW)
        omega: 角频率 (rad/s)
    
    返回:
        温度变化率 (℃/s)
    """
    P = power_function(t, P0, omega)
    return -k * (T - T_env) + alpha * P


def rk4_step(t, T, h, k, T_env, alpha, P0, omega):
    """
    执行一步四阶Runge-Kutta迭代
    
    参数:
        t: 当前时间 (秒)
        T: 当前温度 (℃)
        h: 时间步长 (秒)
        k, T_env, alpha, P0, omega: 系统参数
    
    返回:
        下一时刻的温度
    """
    k1 = dT_dt(t, T, k, T_env, alpha, P0, omega)
    k2 = dT_dt(t + h/2, T + h*k1/2, k, T_env, alpha, P0, omega)
    k3 = dT_dt(t + h/2, T + h*k2/2, k, T_env, alpha, P0, omega)
    k4 = dT_dt(t + h, T + h*k3, k, T_env, alpha, P0, omega)
    
    T_next = T + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)
    return T_next


def simulate_cooling_system(k=0.01, T_env=20, alpha=0.05, P0=100, omega=0.01,
                            T0=25, t_end=5000, h=1.0):
    """
    模拟核动力船舶冷却系统温度变化
    
    参数:
        k: 热交换系数 (1/s)
        T_env: 环境温度 (℃)
        alpha: 功率转换系数 (℃/MW)
        P0: 平均功率 (MW)
        omega: 角频率 (rad/s)
        T0: 初始温度 (℃)
        t_end: 总模拟时间 (秒)
        h: 时间步长 (秒)
    
    返回:
        t_array: 时间数组
        T_array: 温度数组
        P_array: 功率数组
    """
    # 计算步数
    n_steps = int(t_end / h) + 1
    
    # 初始化数组
    t_array = np.zeros(n_steps)
    T_array = np.zeros(n_steps)
    P_array = np.zeros(n_steps)
    
    # 设置初始条件
    t_array[0] = 0
    T_array[0] = T0
    P_array[0] = power_function(0, P0, omega)
    
    # 时间迭代
    for i in range(n_steps - 1):
        t_array[i+1] = t_array[i] + h
        T_array[i+1] = rk4_step(t_array[i], T_array[i], h, k, T_env, alpha, P0, omega)
        P_array[i+1] = power_function(t_array[i+1], P0, omega)
    
    return t_array, T_array, P_array


def plot_results(t_array, T_array, P_array):
    """
    绘制温度和功率随时间变化曲线
    
    参数:
        t_array: 时间数组
        T_array: 温度数组
        P_array: 功率数组
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 绘制温度曲线
    ax1.plot(t_array, T_array, 'r-', linewidth=2)
    ax1.axvline(x=3600, color='k', linestyle='--', linewidth=1.5, label='反应堆停止')
    ax1.set_xlabel('时间 (秒)', fontsize=12)
    ax1.set_ylabel('冷却剂温度 (℃)', fontsize=12)
    ax1.set_title('冷却剂温度随时间变化', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    
    # 标注关键点
    idx_3600 = np.argmin(np.abs(t_array - 3600))
    T_at_3600 = T_array[idx_3600]
    ax1.plot(3600, T_at_3600, 'ro', markersize=10)
    ax1.annotate(f'T(3600s) = {T_at_3600:.2f}℃', 
                xy=(3600, T_at_3600),
                xytext=(3600+200, T_at_3600-50),
                fontsize=10,
                arrowprops=dict(arrowstyle='->', color='red'))
    
    # 绘制功率曲线
    ax2.plot(t_array, P_array, 'b-', linewidth=2)
    ax2.axvline(x=3600, color='k', linestyle='--', linewidth=1.5, label='反应堆停止')
    ax2.set_xlabel('时间 (秒)', fontsize=12)
    ax2.set_ylabel('反应堆功率 (MW)', fontsize=12)
    ax2.set_title('反应堆功率随时间变化', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig('cooling_system_simulation.png', dpi=300, bbox_inches='tight')
    print("温度和功率曲线已保存为 'cooling_system_simulation.png'")
    plt.close()


def analyze_results(t_array, T_array, P_array):
    """
    分析模拟结果
    
    参数:
        t_array: 时间数组
        T_array: 温度数组
        P_array: 功率数组
    """
    print("\n" + "="*60)
    print("结果分析")
    print("="*60)
    
    # 找到特定时刻的温度
    idx_3600 = np.argmin(np.abs(t_array - 3600))
    T_at_3600 = T_array[idx_3600]
    
    print(f"\n关键时刻温度:")
    print(f"  T(0) = {T_array[0]:.4f} ℃ (初始温度)")
    print(f"  T(3600) = {T_at_3600:.4f} ℃ (反应堆停止时)")
    print(f"  T(最终) = {T_array[-1]:.4f} ℃ (模拟结束时)")
    
    # 反应堆运行期间的统计
    mask_running = t_array <= 3600
    T_running = T_array[mask_running]
    print(f"\n反应堆运行期间 (0 ≤ t ≤ 3600s):")
    print(f"  平均温度: {np.mean(T_running):.2f} ℃")
    print(f"  最高温度: {np.max(T_running):.2f} ℃")
    print(f"  最低温度: {np.min(T_running):.2f} ℃")
    print(f"  温度振幅: {(np.max(T_running) - np.min(T_running))/2:.2f} ℃")
    
    # 反应堆停止后的统计
    mask_stopped = t_array > 3600
    T_stopped = T_array[mask_stopped]
    print(f"\n反应堆停止后 (t > 3600s):")
    print(f"  温度衰减: {T_at_3600 - T_array[-1]:.2f} ℃")
    print(f"  衰减速率: 指数衰减，时间常数 τ = 1/k = 100 秒")
    
    # 找到温度首次低于30℃的时刻（反应堆停止后）
    idx_below_30 = np.where((t_array > 3600) & (T_array < 30))[0]
    if len(idx_below_30) > 0:
        t_below_30 = t_array[idx_below_30[0]]
        print(f"  温度首次低于30℃: t = {t_below_30:.0f} 秒")


def parameter_sensitivity_analysis():
    """
    参数敏感性分析
    """
    print("\n" + "="*60)
    print("参数敏感性分析")
    print("="*60)
    
    # 基准参数
    k_base = 0.01
    P0_base = 100
    
    # 改变k
    print("\n1. 改变热交换系数 k 的影响:")
    k_values = [0.005, 0.01, 0.02]
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for k_val in k_values:
        t, T, P = simulate_cooling_system(k=k_val, t_end=5000, h=1.0)
        plt.plot(t, T, linewidth=2, label=f'k = {k_val}')
        
        # 计算理论平衡温度
        T_eq = 20 + (0.05 * P0_base) / k_val
        print(f"  k = {k_val}: 理论平衡温度 = {T_eq:.2f} ℃")
    
    plt.xlabel('时间 (秒)', fontsize=11)
    plt.ylabel('温度 (℃)', fontsize=11)
    plt.title('热交换系数 k 的影响', fontsize=12, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.axvline(x=3600, color='k', linestyle='--', alpha=0.5)
    
    # 改变P0
    print("\n2. 改变平均功率 P0 的影响:")
    P0_values = [50, 100, 150]
    
    plt.subplot(1, 2, 2)
    for P0_val in P0_values:
        t, T, P = simulate_cooling_system(P0=P0_val, t_end=5000, h=1.0)
        plt.plot(t, T, linewidth=2, label=f'P0 = {P0_val} MW')
        
        # 计算理论平衡温度
        T_eq = 20 + (0.05 * P0_val) / k_base
        print(f"  P0 = {P0_val} MW: 理论平衡温度 = {T_eq:.2f} ℃")
    
    plt.xlabel('时间 (秒)', fontsize=11)
    plt.ylabel('温度 (℃)', fontsize=11)
    plt.title('平均功率 P0 的影响', fontsize=12, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.axvline(x=3600, color='k', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('parameter_sensitivity.png', dpi=300, bbox_inches='tight')
    print("\n参数敏感性分析图已保存为 'parameter_sensitivity.png'")
    plt.close()


def main():
    """
    主函数
    """
    print("="*60)
    print("问题3：核动力船舶冷却系统温度模拟（RK4方法）")
    print("="*60)
    
    # 参数设置
    k = 0.01        # 热交换系数 (1/s)
    T_env = 20      # 环境温度 (℃)
    alpha = 0.05    # 功率转换系数 (℃/MW)
    P0 = 100        # 平均功率 (MW)
    omega = 0.01    # 角频率 (rad/s)
    T0 = 25         # 初始温度 (℃)
    t_end = 5000    # 总时间 (秒)
    h = 1.0         # 时间步长 (秒)
    
    print(f"\n系统参数:")
    print(f"  热交换系数 k = {k} 1/s")
    print(f"  环境温度 T_env = {T_env} ℃")
    print(f"  功率转换系数 α = {alpha} ℃/MW")
    print(f"  平均功率 P0 = {P0} MW")
    print(f"  角频率 ω = {omega} rad/s")
    print(f"  初始温度 T0 = {T0} ℃")
    print(f"  模拟时间: 0 - {t_end} 秒")
    print(f"  时间步长 h = {h} 秒")
    
    print(f"\n开始RK4模拟...")
    
    # 执行模拟
    t_array, T_array, P_array = simulate_cooling_system(
        k, T_env, alpha, P0, omega, T0, t_end, h
    )
    
    print(f"模拟完成！共计算 {len(t_array)} 个时间点")
    
    # 绘制结果
    plot_results(t_array, T_array, P_array)
    
    # 分析结果
    analyze_results(t_array, T_array, P_array)
    
    # 参数敏感性分析
    parameter_sensitivity_analysis()
    
    print("\n" + "="*60)
    print("计算完成！")
    print("="*60)


if __name__ == "__main__":
    main()
