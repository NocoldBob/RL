# -*- coding: utf-8 -*-

"""
Project: sanck_sac
Author: fastbiubiu@163.com
Date: 2024/5/9
Description: 
"""
import torch
from environment import SnakeEnv  # 确保这是你的环境
from model import ConvActorCritic  # 确保这是你的模型定义
import numpy as np


def load_model(model, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def play_game(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化环境和模型
    grid_size = 20
    end_score = 80
    env = SnakeEnv(grid_size=grid_size, end_score=end_score)
    input_channels = 3
    output_dim = env.action_space.n

    model = ConvActorCritic(input_channels, output_dim, grid_size).to(device)

    # 加载训练好的模型
    model = load_model(model, model_path)

    state = env.reset()
    done = False

    while not done:
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(device)
        with torch.no_grad():
            action_probs, _ = model(state)
        action = torch.argmax(action_probs).item()  # 选择概率最高的动作

        state, _, done, _ = env.step(action)
        env.render(fps=1000)  # 展示游戏动画


if __name__ == "__main__":
    model_path = 'save_model/best_model.pth'  # 修改为你的模型路径
    # model_path = 'save_model/model_500.pth'  # 修改为你的模型路径
    play_game(model_path)
