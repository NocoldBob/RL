# RL 从零开始：低算力贪吃蛇强化学习

[![CI](https://github.com/NocoldBob/RL/actions/workflows/ci.yml/badge.svg)](https://github.com/NocoldBob/RL/actions/workflows/ci.yml)

面向强化学习新手的中文实践项目。从一个可以直接观察的贪吃蛇环境开始，在普通 CPU
上理解状态、动作、奖励、教师策略、Actor-Critic、DQN、PPO 和独立评估。

项目起点是 2024 年发布的低算力入门教程，重点是让新人快速跑通完整训练闭环。当前版本
沿着这条教学路线继续完善，将可运行实现明确定位为**单步 Actor-Critic**，并补齐状态表达、
行为克隆、独立评估和自动测试。它与 SAC 同属 Actor-Critic 家族；完整 SAC 所需的更多组件，
会放到后续连续动作课程中单独讲解。

第四篇进一步增加了带经验回放和目标网络的 DQN，以及随机策略、教师策略、纯
Actor-Critic、DQN 四种方法的多随机种子统一评估。

第五篇增加了 Rollout、GAE 和裁剪目标组成的 PPO，并将统一基准扩展为五种策略。PPO
评估默认按策略分布进行可复现采样，同时保留贪心动作模式用于对照。

第六篇在完全相同的网络和训练预算下配对比较 DQN 与 Double DQN，并增加初始 Q、实际折扣
回报、目标选择差值和网络动作分歧等诊断，区分价值偏差与最终策略成绩。

第七篇增加 Dueling Q 网络，通过 `2×2` 全因子实验同时比较普通/Dueling 网络结构和普通/
Double TD 目标，并记录状态价值、动作优势、动作间隔、参数量与交互效应。

## 当前版本带来了什么

- 环境严格执行智能体传入的动作，让动作、奖励和下一状态保持一致。
- 启发式规则成为可选的行为克隆教师阶段，教师示范和模型自主训练边界清晰。
- 从旧版 Gym API 迁移到 Gymnasium 的 `terminated` / `truncated` 接口。
- TD 目标不再反向传播到下一状态价值，并加入熵正则和梯度裁剪。
- 训练与评估使用独立环境，支持固定随机种子、CPU/CUDA 和跨平台路径。
- 检查点保存模型、优化器、训练轮次和环境配置，可以恢复训练并直接播放。
- 增加依赖清单、自动测试、GitHub CI 和短训练冒烟测试。
- 增加 DQN、Double DQN、Dueling DQN、PPO、独立播放程序和可导出 JSON/CSV 的基准工具。

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

## DQN 实验

训练默认 DQN：

```powershell
python .\贪吃蛇\train_dqn.py
```

播放 DQN 检查点：

```powershell
python .\贪吃蛇\play_dqn.py .\runs\dqn\checkpoints\best.pt --fps 15
```

第四篇的四策略原始精简数据保存在
[`docs/experiments/04-dqn-benchmark.json`](docs/experiments/04-dqn-benchmark.json)。

## PPO 实验

训练默认 PPO：

```powershell
python .\贪吃蛇\train_ppo.py
```

播放 PPO 检查点：

```powershell
python .\贪吃蛇\play_ppo.py .\runs\ppo\checkpoints\best.pt --fps 15
```

PPO 默认按策略概率采样动作，并固定评测随机数以便复现。使用 `--deterministic` 可以切换为
每一步都选择最大概率动作的对照模式。

用三个训练种子统一比较随机策略、教师策略、纯 Actor-Critic、DQN 和 PPO：

```powershell
python .\贪吃蛇\benchmark.py --seeds 7 42 2026 --episodes 1000 `
  --eval-episodes 100 --device cpu --torch-threads 1
```

第五篇发布前的原始结果保存在
[`docs/experiments/05-ppo-benchmark.json`](docs/experiments/05-ppo-benchmark.json)。

## Double DQN 实验

使用相同配置训练 Double DQN：

```powershell
python .\贪吃蛇\train_dqn.py --double-dqn --output-dir runs\double-dqn
```

配对比较普通 DQN 与 Double DQN，并诊断 Q 值：

```powershell
python .\贪吃蛇\benchmark_double_dqn.py --seeds 7 42 2026 `
  --episodes 1000 --eval-episodes 100 --device cpu --torch-threads 1
```

第六篇原始结果保存在
[`docs/experiments/06-double-dqn-benchmark.json`](docs/experiments/06-double-dqn-benchmark.json)。

## Dueling DQN 实验

只切换为 Dueling 网络结构：

```powershell
python .\贪吃蛇\train_dqn.py --dueling --output-dir runs\dueling-dqn
```

同时启用 Dueling Network 与 Double DQN：

```powershell
python .\贪吃蛇\train_dqn.py --dueling --double-dqn `
  --output-dir runs\dueling-double-dqn
```

使用三个训练种子完成 DQN、Double DQN、Dueling DQN 和 Dueling Double DQN 四组配对实验：

```powershell
python .\贪吃蛇\benchmark_dueling_dqn.py --seeds 7 42 2026 `
  --episodes 1000 --eval-episodes 100 --device cpu --torch-threads 1
```

第七篇原始结果保存在
[`docs/experiments/07-dueling-dqn-benchmark.json`](docs/experiments/07-dueling-dqn-benchmark.json)。

## 可复现验证样例

在 Windows CPU、`seed=42` 的默认小网格配置下，发布前实测 500 Episode 的最后一轮
20 次独立评估结果为：平均奖励 `15.46`、平均吃到 `0.80` 个食物、通关率 `10%`。
这只是流程正确性的固定种子样例，不是算法性能承诺；强化学习结果需要用多个随机种子比较。

## 教学逻辑

### 教师阶段

环境提供 `teacher_action()`，通过安全路径和食物距离给出动作。训练程序显式执行这个动作，
再把示范存入小型回放集，用随机批量交叉熵让 Actor 模仿，并训练 Critic 估计教师轨迹价值。
这个阶段属于行为克隆，随后再切换到 Actor-Critic 自主训练。

### Actor-Critic 阶段

Actor 输出左转、右转和直行三个离散动作的概率；Critic 估计当前状态价值。每一步使用
TD 误差同时更新策略和价值网络，并用少量熵奖励保持探索。

观测包含蛇身、食物、边界和四个蛇头方向通道。方向不能省略：动作是相对方向，当蛇长度
为 1 时，仅凭蛇头位置无法判断左转、右转分别会到哪里。

### 独立评估

评估使用新的环境、固定种子和零教师辅助。Actor-Critic、DQN 使用确定性动作；PPO 按固定
随机数采样策略分布，并可切换贪心模式对照。训练奖励与评估奖励分别记录，避免把启发式规则
的成绩误认为模型能力。

## 项目结构

```text
贪吃蛇/
  environment.py  # Gymnasium 环境与显式教师策略
  model.py        # 卷积 Actor-Critic 与行为克隆更新
  main.py         # 训练、评估、日志和检查点
  play.py         # 无教师辅助的模型可视化
  dqn.py          # Q 网络、经验回放和 DQN 更新
  train_dqn.py    # DQN 训练、评估和检查点
  play_dqn.py     # 无探索的 DQN 可视化
  ppo.py          # Rollout、GAE、PPO Agent 和裁剪更新
  train_ppo.py    # PPO 训练、评估、日志和检查点
  play_ppo.py     # 策略采样或贪心动作的 PPO 可视化
  benchmark.py    # 五种策略的多随机种子统一评估
  benchmark_double_dqn.py # DQN/Double DQN 配对评估和 Q 值诊断
  benchmark_dueling_dqn.py # 四种 DQN 组合的全因子评估和表征诊断
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
- [第二篇：RL 强化学习从小白到老鸟（二）——手撕 GPT](https://blog.csdn.net/bobwww123/article/details/138948884)
- [第三篇：让贪吃蛇训练更稳定、更容易复现](https://blog.csdn.net/bobwww123/article/details/163925583)
- [第三篇 Markdown 源稿](docs/csdn/03-让贪吃蛇训练更稳定更容易复现.md)
- [第四篇：DQN 实战，四种策略同场对照](https://blog.csdn.net/bobwww123/article/details/163925932)
- [第四篇 Markdown 源稿](docs/csdn/04-DQN实战四种策略同场对照.md)
- [第五篇 Markdown：PPO 实战，从单步更新到稳定策略优化](docs/csdn/05-PPO实战从单步更新到稳定策略优化.md)
- [第六篇 Markdown：Double DQN 实战，降低 Q 值成绩就会更好吗](docs/csdn/06-DoubleDQN实战降低Q值成绩就会更好吗.md)
- [第七篇 Markdown：Dueling DQN 实战，先判断局面再选择动作](docs/csdn/07-DuelingDQN实战先判断局面再选择动作.md)

## 后续路线

1. 增加障碍地图和课程难度，比较算法的泛化能力。
2. 增加逐状态的价值与动作优势可视化，帮助理解模型决策。
3. 在连续动作环境中单独讲解完整 SAC。

## License

MIT
