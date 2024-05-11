# -*- coding: utf-8 -*-
"""
Project: 
File Name: environment
Created on: 2024/5/9 14:49
author: fastbiubiu@163.com
Desc: (说明)
"""
import gym
import numpy as np
from gym import spaces
import pygame
from collections import Counter
from collections import deque


class SnakeEnv(gym.Env):
    def __init__(self, grid_size=10, end_score=10):
        self.grid_size = grid_size
        self.end_score = end_score
        self.action_space = spaces.Discrete(3)  # 0: 向左转, 1: 向右转, 2: 直行
        self.observation_space = spaces.Box(low=0, high=1, shape=(3, grid_size, grid_size), dtype=np.float32)
        self.snake = [self._get_center_position()]
        self.food_pos = self._generate_food()
        self.current_direction = np.array([0, 1])  # 初始方向向右
        self.score = 0
        self.visited = set()
        self.last_action = None
        self.reward = 0
        self.visited_positions = []
        self.game_over = False
        self.reset()

    def reset(self):
        self.snake = [self._get_center_position()]
        self.food_pos = self._generate_food()
        self.current_direction = np.array([0, 1])
        self.score = 0
        self.reward = 0
        self.game_over = False
        self.visited.clear()
        self.visited_positions.clear()
        return self._get_observation()

    def _is_safe(self, position):
        if any(position < 0) or any(position >= self.grid_size):
            return False
        if tuple(position) in map(tuple, self.snake[1:]):
            return False
        return True

    def _bfs_safe_path(self, position, depth=11):
        directions = [np.array([-1, 0]), np.array([1, 0]), np.array([0, -1]), np.array([0, 1])]  # 上下左右
        queue = deque([(position, 0)])  # (position, current_depth)
        visited = set(map(tuple, self.snake))  # 蛇身位置被视为已访问

        while queue:
            position, current_depth = queue.popleft()
            if current_depth >= depth:
                return True  # 如果能达到这个深度，认为有安全路径

            for d in directions:
                new_position = position + d
                if tuple(new_position) not in visited and self._is_safe(new_position):
                    queue.append((new_position, current_depth + 1))
                    visited.add(tuple(new_position))

        return False  # 如果搜索完毕没有找到安全路径，则认为不安全

    def step(self, action):
        self.reward = 0
        info = {}
        # 尝试所有可能的动作并选择一个最佳的
        best_action = action
        best_reward = -float('inf')
        #---------------------------------预训练结束后注释本区代码------------------------
        for trial_action in range(3):
            trial_direction = self._get_direction(trial_action)
            trial_head = self.snake[0] + trial_direction
            if self._is_safe(trial_head) and self._bfs_safe_path(trial_head):
                trial_reward = self._calculate_potential_reward(trial_head)
                if trial_reward > best_reward:
                    best_reward = trial_reward
                    best_action = trial_action

            # 执行最佳动作
        self.current_direction = self._get_direction(best_action)
        new_head = self.snake[0] + self.current_direction
        # 检查新的蛇头位置是否超出边界或者撞到自己
        if self._is_collision(new_head) or any(new_head < 0) or any(new_head) >= self.grid_size:
            self.game_over = True
            self.reward -= 5
            return self._get_observation(), self.reward, self.game_over, info
        # --------------------------------------------------------------------------------
        self._calculate_reward(new_head)
        self.last_action = action

        done = self._update_snake_and_food(new_head)

        return self._get_observation(), self.reward, done, info

    def _get_direction(self, action):
        if action == 0:  # 左转
            return np.array([-self.current_direction[1], self.current_direction[0]])
        elif action == 1:  # 右转
            return np.array([self.current_direction[1], -self.current_direction[0]])
        return self.current_direction  # 直行

    def _calculate_potential_reward(self, new_head):
        # 这里简单地使用与食物距离的倒数作为潜在奖励
        # 实际应用中可以根据需要调整奖励的计算方式
        distance_to_food = np.linalg.norm(new_head - self.food_pos)
        return 1 / (distance_to_food + 1)  # 加1避免除以0

    def _calculate_reward(self, new_head):
        distance_before = np.abs(self.snake[0] - self.food_pos).sum()
        distance_after = np.abs(new_head - self.food_pos).sum()

        if distance_after < distance_before:
            self.reward += 0.2  # 靠近食物奖励
        else:
            self.reward -= 0.1

        if tuple(self.snake[0]) not in self.visited:
            self.reward += 0.2  # 探索新格子奖励
            self.visited.add(tuple(self.snake[0]))

        if self._check_repeated_visit():
            self.reward -= 0.4  # 重复访问格子惩罚

    def _update_snake_and_food(self, new_head):
        self.snake.insert(0, new_head)
        if all(new_head == self.food_pos):
            self.score += 1
            self.food_pos = self._generate_food()
            self.reward += 10
        else:
            self.snake.pop()
            self.reward += -0.01

        self.visited_positions.append(new_head)  # 更新访问记录

        if len(self.snake) >= self.end_score:
            self.reward += 1000
            self.game_over = True
            return True
        else:
            return False

    def _check_repeated_visit(self):
        if len(self.visited_positions) < 5:
            return False

        recent_positions = self.visited_positions[-5:]
        visited_counts = Counter(map(tuple, recent_positions))
        for pos, count in visited_counts.items():
            if count >= 2:
                return True
        return False

    def _get_observation(self):
        grid = np.zeros((3, self.grid_size, self.grid_size))
        for segment in self.snake:
            grid[0, tuple(segment)] = 1  # 蛇的位置
        grid[1, tuple(self.food_pos)] = 0.5  # 食物的位置
        grid[2, :, :] = 1  # 边界信息
        grid[2, 1:-1, 1:-1] = 0  # 内部格子为0
        return grid

    def _is_collision(self, position):
        if any(position < 0) or any(position >= self.grid_size):
            return True
        position = tuple(position)
        for part in self.snake[1:]:
            if position == tuple(part):
                return True
        return False

    def _generate_food(self):
        while True:
            food_position = np.random.randint(0, self.grid_size, size=2)
            if not any(np.array_equal(food_position, segment) for segment in self.snake):
                return food_position

    def _get_center_position(self):
        return np.array([self.grid_size // 2, self.grid_size // 2])

    def render(self, mode='human', fps=1000):

        if not hasattr(self, 'screen'):
            pygame.init()
            self.cell_size = 20
            self.screen_width = self.grid_size * self.cell_size
            self.screen_height = self.grid_size * self.cell_size
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            self.clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        self.screen.fill((0, 0, 0))  # 背景设为黑色

        # 绘制网格
        for x in range(0, self.screen_width, self.cell_size):
            for y in range(0, self.screen_height, self.cell_size):
                rect = pygame.Rect(x, y, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, (50, 50, 50), rect, 1)  # 网格颜色较深

        # 绘制食物
        food_x, food_y = self.food_pos
        pygame.draw.rect(self.screen, (0, 255, 0),
                         (food_x * self.cell_size, food_y * self.cell_size, self.cell_size, self.cell_size))

        # 绘制蛇
        # 蛇头
        head_color = (255, 0, 0)  # 蛇头颜色，例如红色
        head_x, head_y = self.snake[0]
        head_rect = (head_x * self.cell_size, head_y * self.cell_size, self.cell_size, self.cell_size)
        self._draw_head(self.screen, head_rect, head_color)

        # 蛇身
        for segment in self.snake[1:]:
            x, y = segment
            pygame.draw.rect(self.screen, (255, 0, 0),
                             (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size))

        # 结束标志
        if self.game_over:
            if len(self.snake) >= self.end_score:
                string = "YOU ARE WIN!"
            else:
                string = "YOU ARE LOST!"
            if mode == 'human':
                print(string)
                # 显示"WIN!"
                font = pygame.font.Font(None, 36)
                text = font.render(string, True, (255, 255, 255))
                text_rect = text.get_rect(center=(self.screen_width // 2, self.screen_height // 2))
                self.screen.blit(text, text_rect)
                # 停止游戏更新，但保持渲染
                while True:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            return
                    self.clock.tick(fps)  # 控制帧率
                    pygame.display.flip()
        else:
            pygame.display.flip()
            self.clock.tick(fps)  # 控制帧率

    def _draw_head(self, screen, rect, color):
        # 绘制圆形蛇头
        pygame.draw.circle(screen, color, (rect[0] + self.cell_size // 2, rect[1] + self.cell_size // 2),
                           self.cell_size // 2)

        # 根据蛇头方向绘制眼睛
        eye_color = (0, 0, 0)  # 眼睛颜色，例如黑色
        if self.current_direction[0] > 0:  # 向右
            eye_pos = (rect[0] + self.cell_size // 4, rect[1] + self.cell_size // 4)
        elif self.current_direction[0] < 0:  # 向左
            eye_pos = (rect[0] + 3 * self.cell_size // 4, rect[1] + self.cell_size // 4)
        elif self.current_direction[1] > 0:  # 向下
            eye_pos = (rect[0] + self.cell_size // 4, rect[1] + 3 * self.cell_size // 4)
        else:  # 向上
            eye_pos = (rect[0] + 3 * self.cell_size // 4, rect[1] + 3 * self.cell_size // 4)

        # 绘制两个眼睛
        pygame.draw.circle(screen, eye_color, eye_pos, self.cell_size // 8)
        pygame.draw.circle(screen, eye_color, (eye_pos[0], eye_pos[1] + self.cell_size // 2), self.cell_size // 8)

    def close(self):
        if hasattr(self, 'screen'):
            pygame.quit()
