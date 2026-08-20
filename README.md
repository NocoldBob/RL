# RL 从零开始：低算力贪吃蛇 Actor-Critic

[![CI](https://github.com/NocoldBob/RL/actions/workflows/ci.yml/badge.svg)](https://github.com/NocoldBob/RL/actions/workflows/ci.yml)

面向强化学习新手的中文实践项目。从一个可以直接观察的贪吃蛇环境开始，在普通 CPU
上理解状态、动作、奖励、教师策略、Actor-Critic 和独立评估。

> 2024 年的初版 README 和配套文章把模型称为 SAC。当前实现已经更正为
> **单步 Actor-Critic**：它没有 SAC 所需的双 Q 网络、目标网络、经验回放和最大熵目标。

## 这次修正了什么

- 环境严格执行智能体传入的动作，不再偷偷替换成启发式动作。
- 启发式规则改成可选的行为克隆教师阶段，教师动作和实际动作始终一致。
- 从旧版 Gym API 迁移到 Gymnasium 的 `terminated` / `truncated` 接口。
- TD 目标不再反向传播到下一状态价值，并加入熵正则和梯度裁剪。
- 训练与评估使用独立环境，支持固定随机种子、CPU/CUDA 和跨平台路径。
- 检查点保存模型、优化器、训练轮次和环境配置，可以恢复训练并直接播放。
- 增加依赖清单、自动测试、GitHub CI 和短训练冒烟测试。

## 环境要求

- Python 3.10+
- Windows、Linux 或 macOS
- CUDA 可选；小网格实验使用 CPU 即可

## 快速开始

```powershell
git clone https://github.com/NocoldBob/RL.git
cd RL
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Linux/macOS 激活环境：

```bash
source .venv/bin/activate
```

先运行一个几十秒内可完成的流程检查：

```powershell
python .\贪吃蛇\main.py --episodes 20 --teacher-episodes 5 --grid-size 6 `
  --end-score 4 --max-steps 100 --eval-interval 10 --eval-episodes 3
```

开始默认教学实验：

```powershell
python .\贪吃蛇\main.py
```

默认实验使用 `6x6` 网格、训练 `500` 个 Episode；前 `100` 个 Episode 使用教师策略进行
小型示范回放和批量行为克隆，之后完全由 Actor-Critic 选择并执行动作。需要从零开始纯
强化学习时：

```powershell
python .\贪吃蛇\main.py --teacher-episodes 0
```

## 查看训练

训练产物默认保存在 `runs/snake`：

- `checkpoints/best.pt`：独立评估奖励最高的模型
- `checkpoints/latest.pt`：训练结束时的模型
- `summary.json`：本次训练摘要
- `tensorboard/`：奖励、得分、损失和评估曲线

启动 TensorBoard：

```powershell
tensorboard --logdir .\runs\snake\tensorboard
```

播放最佳模型：

```powershell
python .\贪吃蛇\play.py .\runs\snake\checkpoints\best.pt --fps 15
```

播放阶段不会调用教师策略，画面中的每个动作都来自模型。

## 可复现验证样例

在 Windows CPU、`seed=42` 的默认小网格配置下，发布前实测 500 Episode 的最后一轮
20 次独立评估结果为：平均奖励 `15.46`、平均吃到 `0.80` 个食物、通关率 `10%`。
这只是流程正确性的固定种子样例，不是算法性能承诺；强化学习结果需要用多个随机种子比较。

## 教学逻辑

### 教师阶段

环境提供 `teacher_action()`，通过安全路径和食物距离给出动作。训练程序显式执行这个动作，
再把示范存入小型回放集，用随机批量交叉熵让 Actor 模仿，并训练 Critic 估计教师轨迹价值。
它属于行为克隆，不冒充强化学习更新。

### Actor-Critic 阶段

Actor 输出左转、右转和直行三个离散动作的概率；Critic 估计当前状态价值。每一步使用
TD 误差同时更新策略和价值网络，并用少量熵奖励保持探索。

观测包含蛇身、食物、边界和四个蛇头方向通道。方向不能省略：动作是相对方向，当蛇长度
为 1 时，仅凭蛇头位置无法判断左转、右转分别会到哪里。

### 独立评估

评估使用新的环境、固定种子、确定性动作和零教师辅助。训练奖励与评估奖励分别记录，
避免把启发式规则的成绩误认为模型能力。

## 项目结构

```text
贪吃蛇/
  environment.py  # Gymnasium 环境与显式教师策略
  model.py        # 卷积 Actor-Critic 与行为克隆更新
  main.py         # 训练、评估、日志和检查点
  play.py         # 无教师辅助的模型可视化
tests/            # 环境、模型、检查点和短训练测试
docs/csdn/        # 可发布到 CSDN 的后续教程
```

## 测试

```powershell
python -m pip install -r requirements-dev.txt
python -m ruff check .
python -m ruff format --check .
python -m pytest
```

## 教程

- [第一篇：RL 强化学习从小白到老鸟（一）——速通贪吃蛇游戏](https://blog.csdn.net/bobwww123/article/details/138722671)
- [第二篇 Markdown：先把贪吃蛇训练做对](docs/csdn/02-先把贪吃蛇训练做对.md)

## 后续路线

1. 增加随机策略和教师策略基线对比。
2. 增加适合离散动作的 DQN 课程。
3. 增加 A2C/PPO 与多随机种子实验。
4. 在连续动作环境中单独讲解真正的 SAC。

## License

MIT
