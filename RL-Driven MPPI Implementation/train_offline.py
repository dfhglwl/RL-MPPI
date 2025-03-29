import os
import numpy as np
import torch
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

from environment.uav import Quadrotor
from models.dsac import DSAC
from utils.buffer import ReplayBuffer
from utils.utils import create_directory, set_seed, evaluate_policy, visualize_trajectory


def parse_args():
    parser = argparse.ArgumentParser()
    # 训练参数
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu", type=str)
    parser.add_argument("--save_dir", default="./results", type=str)
    parser.add_argument("--eval_freq", default=5000, type=int)
    parser.add_argument("--eval_episodes", default=10, type=int)
    parser.add_argument("--max_steps", default=500000, type=int)

    # 环境参数
    parser.add_argument("--dt", default=0.05, type=float)

    # DSAC参数
    parser.add_argument("--hidden_dim", default=256, type=int)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--tau", default=0.005, type=float)
    parser.add_argument("--lr_actor", default=3e-5, type=float)
    parser.add_argument("--lr_critic", default=1e-4, type=float)
    parser.add_argument("--buffer_size", default=1e6, type=int)
    return parser.parse_args()


def main():
    # 解析参数
    args = parse_args()

    # 创建保存目录
    create_directory(args.save_dir)
    create_directory(os.path.join(args.save_dir, "models"))
    create_directory(os.path.join(args.save_dir, "plots"))

    # 设置随机种子
    set_seed(args.seed)

    # 创建环境
    env = Quadrotor()
    env.dt = args.dt

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
        gamma=args.gamma,
        tau=args.tau,
        hidden_dim=args.hidden_dim,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        auto_alpha=True
    )

    # 创建经验回放缓冲区
    memory = ReplayBuffer(size=args.buffer_size)

    # 训练循环
    state = env.reset()
    episode_reward = 0
    episode_steps = 0
    episode_num = 0

    rewards = []
    eval_rewards = []
    success_rates = []

    for step in tqdm(range(args.max_steps), desc="Training"):
        # 选择动作
        action = agent.select_action(state)

        # 执行动作
        next_state, reward, done, _, _ = env.step(state, action)

        # 存储转换
        memory.add(state, action, reward, next_state, float(done))

        state = next_state
        episode_reward += reward
        episode_steps += 1

        # 更新代理
        if len(memory) > args.batch_size:  # 使用魔术方法__len__
            agent.update_parameters(memory, args.batch_size)

        # 如果回合结束
        if done:
            rewards.append(episode_reward)
            print(f"Episode {episode_num}: reward={episode_reward}, steps={episode_steps}")

            # 重置环境
            state = env.reset()
            episode_reward = 0
            episode_steps = 0
            episode_num += 1

        # 定期评估
        if (step + 1) % args.eval_freq == 0:
            eval_result = evaluate_policy(agent, env, args.eval_episodes)
            eval_rewards.append(eval_result["avg_reward"])
            success_rates.append(eval_result["success_rate"])

            print(f"Step {step + 1}: Evaluation - Avg Reward: {eval_result['avg_reward']:.2f}, "
                  f"Success Rate: {eval_result['success_rate']:.2f}, "
                  f"Completion Time: {eval_result['avg_completion_time']:.2f}s")

            # 可视化一个轨迹
            state_eval = env.reset()
            states = [state_eval]
            done = False

            while not done:
                action = agent.select_action(state_eval, evaluate=True)
                next_state, _, done, _, _ = env.step(state_eval, action)
                states.append(next_state)
                state_eval = next_state
                if len(states) > 200:  # 防止无限循环
                    break

            visualize_trajectory(
                np.array(states),
                target=[0, 0, 0],
                save_path=os.path.join(args.save_dir, "plots", f"trajectory_{step + 1}.png")
            )

            # 保存代理
            agent.save(os.path.join(args.save_dir, "models"), iteration=step + 1)

    # 绘制训练曲线
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(rewards)
    plt.title("Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")

    plt.subplot(2, 1, 2)
    eval_steps = np.arange(0, args.max_steps + 1, args.eval_freq)[1:len(eval_rewards) + 1]
    plt.plot(eval_steps, success_rates)
    plt.title("Success Rate")
    plt.xlabel("Step")
    plt.ylabel("Success Rate")

    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "training_curves.png"))

    # 保存最终模型
    agent.save(os.path.join(args.save_dir, "models"))

    print("Training completed!")


if __name__ == "__main__":
    main()