import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import torch

def create_directory(directory):
    """创建目录（如果不存在）"""
    import os
    if not os.path.exists(directory):
        os.makedirs(directory)

def set_seed(seed):
    """设置随机种子以确保可重复性"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def visualize_trajectory(states, target=None, target_radius=1.5, save_path=None, elev=30, azim=45):
    """
    可视化UAV轨迹
    
    Args:
        states: 轨迹状态数组
        target: 目标位置
        target_radius: 目标区域半径
        save_path: 保存路径
        elev: 视图仰角 (默认30度)
        azim: 视图方位角 (默认45度)
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 设置更好的视角
    ax.view_init(elev=elev, azim=azim)
    
    # 提取位置
    positions = states[:, :3]
    
    # 绘制轨迹
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=2)
    
    # 标记起点和终点
    ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], c='g', marker='o', s=100, label='Start')
    ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], c='r', marker='*', s=100, label='End')
    
    # 绘制目标（如果有）
    if target is not None:
        ax.scatter(target[0], target[1], target[2], c='y', marker='X', s=100, label='Target')
        
        # 绘制目标区域（使用参数定义的半径）
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        radius = target_radius
        x = radius * np.cos(u) * np.sin(v) + target[0]
        y = radius * np.sin(u) * np.sin(v) + target[1]
        z = radius * np.cos(v) + target[2]
        ax.plot_wireframe(x, y, z, color='y', alpha=0.1)
    
    # 设置轴标签和标题
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title('UAV Trajectory')
    
    # 设置坐标轴范围
    ax.set_xlim([-10, 10])
    ax.set_ylim([-10, 10])
    ax.set_zlim([-10, 10])
    
    ax.legend()
    
    # 添加网格以增强深度感知
    ax.grid(True)
    
    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

def evaluate_policy(agent, env, eval_episodes=10, target_radius=1.5, render=False, normalize_fn=None):
    """评估策略的性能"""
    avg_reward = 0.0
    success_rate = 0.0
    completion_times = []
    
    for episode in range(eval_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        while not done:
            # 应用状态归一化（如果提供）
            if normalize_fn is not None:
                normalized_state = normalize_fn(state)
                action = agent.select_action(normalized_state, evaluate=True)
            else:
                action = agent.select_action(state, evaluate=True)
                
            next_state, reward, done, info = env.step(state, action)
            
            # 从info字典中获取需要的信息
            position = info.get('position', None)
            distance = info.get('distance', 0)
            
            state = next_state
            episode_reward += reward
            step += 1
            
            # 使用info中的distance
            if distance <= target_radius:
                success_rate += 1
                completion_times.append(step * 0.05)  # 乘以时间步长
                
            if render:
                # 可视化当前状态
                pass
                
            if step >= 1000:  # 防止无限循环
                break
        
        avg_reward += episode_reward
    
    # 计算平均值
    avg_reward /= eval_episodes
    success_rate /= eval_episodes
    avg_completion_time = np.mean(completion_times) if completion_times else float('inf')
    
    return {
        'avg_reward': avg_reward,
        'success_rate': success_rate,
        'avg_completion_time': avg_completion_time
    }