import os
import numpy as np
import torch
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

from environment.uav import Quadrotor
from models.dynamics import DynamicsModel, DynamicsTrainer
from models.dsac import DSAC
from utils.buffer import ReplayBuffer
from utils.utils import create_directory, set_seed, visualize_trajectory

def parse_args():
    parser = argparse.ArgumentParser()
    # 训练参数
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str)
    parser.add_argument("--save_dir", default="./results", type=str)
    parser.add_argument("--rl_model_dir", default="./results/models", type=str)
    parser.add_argument("--collect_steps", default=50000, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--batch_size", default=256, type=int)
    
    # 环境参数
    parser.add_argument("--dt", default=0.1, type=float)
    
    # 动力学模型参数
    parser.add_argument("--hidden_dim", default=256, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--buffer_size", default=1e6, type=int)
    
    return parser.parse_args()

def main():
    # 解析参数
    args = parse_args()
    
    # 创建保存目录
    create_directory(args.save_dir)
    create_directory(os.path.join(args.save_dir, "dynamics_models"))
    create_directory(os.path.join(args.save_dir, "plots"))
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建环境
    env = Quadrotor()
    env.dt = args.dt
    
    # 状态和动作维度
    state_dim = 13
    action_dim = 4
    
    # 创建DSAC代理（用于收集数据）
    agent = DSAC(
        state_dim=state_dim,
        action_dim=action_dim,
        device=args.device,
        hidden_dim=args.hidden_dim,
    )
    
    # 加载预训练的RL模型（如果存在）
    try:
        agent.load(args.rl_model_dir)
        print("Loaded pre-trained RL model.")
    except:
        print("No pre-trained RL model found. Using randomly initialized policy.")
    
    # 创建经验回放缓冲区
    memory = ReplayBuffer(size=args.buffer_size)
    
    # 收集动力学数据
    print("Collecting dynamics data...")
    state = env.reset()
    
    for step in tqdm(range(args.collect_steps), desc="Collecting Data"):
        # 选择动作（加入噪声以增加探索）
        if step < args.collect_steps // 3:
            # 前1/3的步骤使用完全随机动作
            action = np.random.uniform(
                low=np.array([0.0, -5.0, -5.0, -5.0]),
                high=np.array([20.0, 5.0, 5.0, 5.0]),
                size=(action_dim,)
            )
        else:
            # 后2/3的步骤使用策略加噪声
            action = agent.select_action(state)
            action += np.random.normal(0, 0.3, size=action_dim)
            action = np.clip(
                action,
                np.array([0.0, -5.0, -5.0, -5.0]),
                np.array([20.0, 5.0, 5.0, 5.0])
            )
        
        # 执行动作
        next_state, reward, done, _, _ = env.step(state, action)
        
        # 存储转换
        memory.add(state, action, reward, next_state, float(done))
        
        if done:
            state = env.reset()
        else:
            state = next_state
    
    print(f"Collected {memory.size()} transitions.")
    
    # 创建并训练动力学模型
    dynamics_trainer = DynamicsTrainer(
        state_dim=state_dim,
        action_dim=action_dim,
        device=args.device,
        lr=args.lr,
        hidden_dim=args.hidden_dim
    )
    
    # 训练动力学模型
    print("Training dynamics model...")
    losses = []
    
    for epoch in range(args.epochs):
        # 训练一个轮次
        train_info = dynamics_trainer.train_epochs(memory, epochs=1, batch_size=args.batch_size)
        losses.append(train_info["avg_dynamics_loss"])
        
        print(f"Epoch {epoch+1}/{args.epochs}: Loss = {train_info['avg_dynamics_loss']:.6f}")
        
        # 每10个轮次保存一次
        if (epoch + 1) % 10 == 0:
            dynamics_trainer.save(os.path.join(args.save_dir, "dynamics_models"), iteration=epoch+1)
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title("Dynamics Model Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True)
    plt.savefig(os.path.join(args.save_dir, "plots", "dynamics_training_loss.png"))
    
    # 保存最终模型
    dynamics_trainer.save(os.path.join(args.save_dir, "dynamics_models"))
    
    # 测试动力学模型
    print("Testing dynamics model...")
    state = env.reset()
    
    # 生成一条预测轨迹
    actions = []
    true_states = [state]
    
    for _ in range(100):
        action = agent.select_action(state, evaluate=True)
        actions.append(action)
        
        next_state, _, done, _, _ = env.step(state, action)
        true_states.append(next_state)
        
        state = next_state
        
        if done:
            break
    
    # 使用动力学模型预测相同动作序列
    predicted_states = dynamics_trainer.model.predict_trajectory(true_states[0], np.array(actions))
    
    # 可视化真实轨迹和预测轨迹
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    true_states_array = np.array(true_states)
    predicted_states_array = np.array(predicted_states)
    
    ax.plot(true_states_array[:, 0], true_states_array[:, 1], true_states_array[:, 2], 'b-', linewidth=2, label='True')
    ax.plot(predicted_states_array[:, 0], predicted_states_array[:, 1], predicted_states_array[:, 2], 'r--', linewidth=2, label='Predicted')
    
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title('True vs Predicted Trajectory')
    ax.legend()
    
    plt.savefig(os.path.join(args.save_dir, "plots", "dynamics_prediction_test.png"))
    
    print("Dynamics model training completed!")

if __name__ == "__main__":
    main()
