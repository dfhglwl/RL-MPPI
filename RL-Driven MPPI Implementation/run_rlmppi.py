import os
import numpy as np
import torch
import argparse
import time
import matplotlib.pyplot as plt

from environment.uav import Quadrotor
from models.dsac import DSAC
from models.dynamics import DynamicsModel
from models.mppi import MPPIController
from utils.utils import create_directory, set_seed, visualize_trajectory

def parse_args():
    parser = argparse.ArgumentParser()
    # 运行参数
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str)
    parser.add_argument("--save_dir", default="./results/rlmppi", type=str)
    parser.add_argument("--rl_model_dir", default="./results/models", type=str)
    parser.add_argument("--dynamics_model_dir", default="./results/dynamics_models", type=str)
    parser.add_argument("--num_episodes", default=10, type=int)
    parser.add_argument("--max_steps", default=1000, type=int)  # 最大回合长度为1000
    
    # 环境参数
    parser.add_argument("--dt", default=0.05, type=float)  # 时间步长0.05秒
    parser.add_argument("--target_radius", default=1.5, type=float)  # 目标区域半径1.5米
    
    # MPPI参数 - 论文中的标准参数
    parser.add_argument("--K", default=6, type=int, help="MPPI迭代次数")
    parser.add_argument("--N", default=15, type=int, help="预测时域长度")
    parser.add_argument("--num_samples", default=125, type=int, help="每次迭代的样本数")
    parser.add_argument("--n_rl", default=125, type=int, help="RL引导样本数")
    parser.add_argument("--n_mppi", default=125, type=int, help="MPPI样本数")
    parser.add_argument("--sigma_min", default=0.5, type=float, help="最小标准差")
    parser.add_argument("--lambda_", default=1.0, type=float, help="MPPI温度参数")
    
    # 消融研究参数
    parser.add_argument("--method", default="rlmppi", type=str, 
                        choices=["rlmppi", "mppi", "rl"], help="控制方法选择")
    
    return parser.parse_args()

def main():
    # 解析参数
    args = parse_args()
    
    # 创建保存目录
    create_directory(args.save_dir)
    create_directory(os.path.join(args.save_dir, "trajectories"))
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建环境
    env = Quadrotor()
    env.dt = args.dt
    # 设置环境权重参数
    env.wr = 0.02   # 位置误差权重
    env.wv = 0.01   # 速度误差权重
    env.wq = 0.001  # 姿态误差权重
    env.ww = 0.001  # 角速度误差权重
    
    # 状态和动作维度
    state_dim = 13
    action_dim = 4
    
    # 动作限制
    action_low = np.array([0.0, -5.0, -5.0, -5.0])
    action_high = np.array([20.0, 5.0, 5.0, 5.0])
    
    # 创建DSAC代理
    agent = DSAC(
        state_dim=state_dim,
        action_dim=action_dim,
        device=args.device,
        gamma=0.99,            # 折扣因子
        tau=0.005,             # 目标网络软更新系数
        lr_actor=2e-5,         # 策略网络学习率
        lr_critic=1e-5,        # 值函数网络学习率
        lr_alpha=1e-5,         # 温度系数α的学习率
        target_entropy=-4,     # 策略熵的目标值
    )
    
    # 加载RL模型
    agent.load(args.rl_model_dir)
    print("Loaded RL model.")
    
    # 创建并加载动力学模型
    dynamics_model = DynamicsModel(state_dim, action_dim).to(args.device)
    dynamics_model.load_state_dict(torch.load(os.path.join(args.dynamics_model_dir, "dynamics.pth")))
    print("Loaded dynamics model.")
    
    # 创建MPPI控制器 - 移除adaptive_sigma参数
    if args.method in ["rlmppi", "mppi"]:
        mppi_controller = MPPIController(
            state_dim=state_dim,
            action_dim=action_dim,
            actor=agent.actor,
            critic=agent.critic,
            dynamics_model=dynamics_model,
            device=args.device,
            K=args.K,
            N=args.N,
            lambda_=args.lambda_,
            num_samples=args.num_samples,
            n_rl=args.n_rl if args.method == "rlmppi" else 0,  # RL引导样本数
            n_mppi=args.n_mppi,                                # MPPI样本数
            sigma_min=args.sigma_min,                          # 最小标准差
            action_low=action_low,
            action_high=action_high,
        )
    
    # 评估统计
    episode_rewards = []
    success_count = 0
    episode_lengths = []
    computation_times = []
    
    # 运行多个回合
    for episode in range(args.num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_steps = 0
        states = [state]
        velocities = []
        
        for step in range(args.max_steps):
            # 选择动作
            start_time = time.time()
            
            if args.method == "rlmppi":
                # RL-MPPI混合控制
                action = mppi_controller.control(state)
            elif args.method == "mppi":
                # 纯MPPI控制
                action = mppi_controller.control(state)
            elif args.method == "rl":
                # 纯RL控制
                action = agent.select_action(state, evaluate=True)
            
            computation_time = time.time() - start_time
            computation_times.append(computation_time)
            
            # 执行动作
            next_state, reward, done, position, dist = env.step(state, action)
            
            # 记录速度数据进行分析
            velocities.append(np.linalg.norm(next_state[3:6]))  # 速度向量范数
            
            state = next_state
            states.append(state)
            episode_reward += reward
            episode_steps += 1
            
            if done:
                # 使用目标半径为1.5判断成功
                dist_actual = np.sqrt(dist)  # env.cost_r_I是平方距离
                if dist_actual <= args.target_radius:  # 成功达到目标
                    success_count += 1
                break
                
            if step >= args.max_steps - 1:
                break
        
        # 回合统计
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_steps)
        
        dist_actual = np.sqrt(dist)
        avg_velocity = np.mean(velocities) if velocities else 0
        max_velocity = np.max(velocities) if velocities else 0
        final_velocity = velocities[-1] if velocities else 0
        
        print(f"Episode {episode+1}: Reward={episode_reward:.2f}, "
              f"Steps={episode_steps}, {'Success' if dist_actual <= args.target_radius else 'Failure'}, "
              f"Avg Vel={avg_velocity:.2f}, Max Vel={max_velocity:.2f}, Final Vel={final_velocity:.2f}")
        
        # 可视化轨迹
        visualize_trajectory(
            np.array(states), 
            target=[0, 0, 0],
            target_radius=args.target_radius,
            save_path=os.path.join(args.save_dir, "trajectories", f"episode_{episode+1}.png")
        )
    
    # 汇总统计
    success_rate = success_count / args.num_episodes
    avg_reward = np.mean(episode_rewards)
    avg_steps = np.mean(episode_lengths)
    avg_computation_time = np.mean(computation_times) * 1000  # 转换为毫秒
    
    print("\n=== 性能统计 ===")
    print(f"方法: {args.method}")
    print(f"成功率: {success_rate:.2f}")
    print(f"平均奖励: {avg_reward:.2f}")
    print(f"平均步数: {avg_steps:.2f}")
    print(f"平均计算时间: {avg_computation_time:.2f} ms")
    
    # 保存统计结果
    stats = {
        "method": args.method,
        "success_rate": success_rate,
        "avg_reward": avg_reward,
        "avg_steps": avg_steps,
        "avg_computation_time": avg_computation_time,
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "computation_times": computation_times,
    }
    
    np.save(os.path.join(args.save_dir, f"{args.method}_stats.npy"), stats)
    
    print(f"结果已保存至 {args.save_dir}")

if __name__ == "__main__":
    main()
