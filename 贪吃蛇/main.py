# -*- coding: utf-8 -*-
"""
Project: 
File Name: main
Created on: 2024/5/9 14:47
author: fastbiubiu@163.com
Desc: (说明)
"""
import torch
from torch.utils.tensorboard import SummaryWriter
import os
from environment import SnakeEnv
from model import ConvActorCritic


def early_stopping_criteria(rewards, patience=3):
    """
    早停准则函数
    Args:
    - rewards: 模型在验证集上的奖励历史
    - patience: 允许模型在验证集上性能不提升的最大连续周期数
    Returns:
    - 是否触发早停
    """
    if len(rewards) < patience:
        return False  # 不触发早停，因为还没有达到最大连续周期数

    recent_rewards = rewards[-patience:]  # 取最近的patience个奖励值
    best_reward = max(rewards)  # 历史最佳奖励

    if all(reward <= best_reward for reward in recent_rewards):
        return True  # 触发早停，因为连续patience个周期内性能没有提升
    else:
        return False  # 不触发早停


def evaluate_model(agent, env, device, num_eval_episodes):
    rewards = []  # 存储每个评估周期的奖励值
    for _ in range(num_eval_episodes):
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(device)
        done = False
        episode_reward = 0
        step_count = 0
        while not done and step_count <= max_steps:
            action_probs, _ = agent(state)
            action = torch.argmax(action_probs).item()  # 选择概率最高的动作
            next_state, reward, done, _ = env.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float).unsqueeze(0).to(device)
            episode_reward += reward
            state = next_state.to(device)
            if done:
                break
            step_count += 1
        rewards.append(episode_reward)  # 将每个评估周期的奖励值添加到列表中

    avg_reward = sum(rewards) / num_eval_episodes
    return avg_reward, rewards  # 返回平均奖励值和奖励值列表


if __name__ == '__main__':
    # 检查是否有可用的CUDA设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = r"./save_model"
    log_dir = r"./logs"
    grid_size = 20  # 或其他你的环境设置的值
    end_score = 60  # 游戏结束条件
    env = SnakeEnv(grid_size=grid_size, end_score=end_score)
    max_episodes = 10000000
    max_steps = 2000
    eval_freq = 1000  # 评估评率
    num_eval_episodes = 50  # 评估周期
    save_interval = 500  # 固定保存频率
    lr = 1e-4
    weight_decay = 1e-5
    gamma = 0.99
    alpha = 0.2
    beta = 1

    input_channels = 3  # Assuming the observation space is a single-channel image
    output_dim = env.action_space.n
    agent = ConvActorCritic(input_channels, output_dim, grid_size, lr, weight_decay).to(device)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    load_model_path = f"{model_path}\\model_2500.pth"
    if os.path.exists(load_model_path):
        agent.load_model(agent.to(device), filename=load_model_path)
        print(f"Loaded model from {load_model_path}")
    else:
        print("No model to load. Starting a new training session.")

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir)
    best_reward = float('-inf')  # 初始化最佳奖励值为负无穷
    for episode in range(2501, max_episodes):
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(device)  # # 为了输入卷积，增加一个维度（卷积4维）
        # print(state)
        step_count = 0
        episode_reward = 0
        done = False

        while not done and step_count < max_steps:
            action_probs, value = agent(state)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()

            next_state, reward, done, _ = env.step(action.item())
            next_state = torch.tensor(next_state, dtype=torch.float).unsqueeze(0).to(device)

            # 智能体根据奖励函数更新策略
            agent.update(state, action, reward, next_state, done, gamma)
            episode_reward += reward
            state = next_state.to(device)
            if done:
                break
            step_count += 1
            # env.render(mode="train",fps=100)  # 运行可视化

        writer.add_scalar('Reward', round(episode_reward, 5), episode)
        writer.add_scalar('Steps', step_count, episode)  # 记录步长

        print(f"Episode {episode}, Reward: {round(episode_reward, 3)}, step_count: {step_count}")
        if episode % eval_freq == 0 and episode != 0:
            avg_reward, rewards = evaluate_model(agent, env, device, num_eval_episodes)

            # 更新最佳模型
            if avg_reward > best_reward:
                best_reward = avg_reward
                agent.save_model(agent.to(device), filename=f"{model_path}\\best_model.pth")

            # 早停逻辑
            # if early_stopping_criteria(rewards, patience=30):  # patience 连续多个周期没有表现提升
            #     print("Early stopping triggered.")
            #     break

        # 保存周期性模型
        if episode % save_interval == 0:
            agent.save_model(agent.to(device), filename=f"{model_path}\\model_{episode}.pth")
    env.close()  # 结束游戏运行
    writer.close()
