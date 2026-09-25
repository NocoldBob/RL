# RL 从零开始：低算力强化学习实践

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

第八篇不再增加算法，而是让六个模型读取四个完全相同的固定局面，检查动作选择、立即碰撞
风险、策略概率、Q 值以及 Dueling Network 的状态价值与动作优势，完成离散控制阶段复盘。

第九篇使用连续山地车开启新阶段，把动作从三个固定选项扩展为 `[-1, 1]` 连续油门，并通过
四种规则基线比较成功率、速度与动作成本，为连续 PPO 和 SAC 建立评估基准。

第十篇将 PPO 改为带 `tanh` 压缩和概率修正的高斯策略，使用三个训练种子学习连续油门。
实验同时保留两个成功模型和一个零油门局部最优，展示连续 PPO 的能力与随机种子敏感性。

第十一篇引入 SAC 的经验回放、双 Q 网络、目标网络软更新和自动温度参数。实验发现逐步独立
随机动作无法在惯性任务中找到终点，改用保持 10 步的时间相关随机探索后，三个训练种子的
最终测试成功率分别达到 `95%`、`100%` 和 `100%`。

第十二篇读取 PPO 与 SAC 的原始实验报告进行配对比较。相同 5 万环境步下，PPO 在成功种子上
学得更早且动作成本更低，SAC 最终对三个训练种子都可靠、到达终点更快，但训练计算量更大。

第十三篇结束连续山地车阶段，进入 `LunarLanderContinuous-v3`。课程先拆解 8 维状态、主/侧
发动机的二维连续动作与死区，并用零推力、随机控制、规则控制和带噪规则建立训练前基线。

第十四篇把 PPO 与 SAC 迁移到二维动作后保留失败结果。共享网络 PPO 即使训练 100 万步仍未
稳定着陆；拆分 Actor/Critic 后一度达到 `55%` 着陆率，随后再次崩塌，由此转入可观测的稳定化
实验，而不是继续盲目增加训练步数。

## 当前版本带来了什么

- 环境严格执行智能体传入的动作，让动作、奖励和下一状态保持一致。
- 启发式规则成为可选的行为克隆教师阶段，教师示范和模型自主训练边界清晰。
- 从旧版 Gym API 迁移到 Gymnasium 的 `terminated` / `truncated` 接口。
- TD 目标不再反向传播到下一状态价值，并加入熵正则和梯度裁剪。
- 训练与评估使用独立环境，支持固定随机种子、CPU/CUDA 和跨平台路径。
- 检查点保存模型、优化器、训练轮次和环境配置，可以恢复训练并直接播放。
- 增加依赖清单、自动测试、GitHub CI 和短训练冒烟测试。
- 增加 DQN、Double DQN、Dueling DQN、PPO、独立播放程序和可导出 JSON/CSV 的基准工具。
- 增加固定局面决策检查，可生成机器可读报告和教程配图。
- 增加连续山地车环境的四种基线、配对评估、轨迹记录与可视化。
- 增加连续动作 PPO、正确的截断自举、多训练种子评估和规则基线对照。
- 增加 SAC、经验回放、双 Q 网络、自动温度调整和时间相关探索实验。
- 增加 LunarLander 二维连续动作环境、规则基线、着陆指标和轨迹可视化。

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

## 固定局面决策检查

第八篇使用第 5 篇和第 7 篇生成的 `seed=42` 检查点。若尚未训练，先运行：

```powershell
python .\贪吃蛇\benchmark.py --seeds 7 42 2026 --episodes 1000 `
  --eval-episodes 100 --device cpu --torch-threads 1 `
  --output-dir runs\benchmark-ppo
python .\贪吃蛇\benchmark_dueling_dqn.py --seeds 7 42 2026 `
  --episodes 1000 --eval-episodes 100 --device cpu --torch-threads 1
```

然后让六个模型读取四个固定状态并生成报告与配图：

```powershell
python .\贪吃蛇\inspect_decisions.py
```

原始决策报告保存在
[`docs/experiments/08-decision-inspection.json`](docs/experiments/08-decision-inspection.json)。固定状态
检查用于解释具体选择，不替代多随机种子的完整游戏评估。

## 连续控制入门

第九篇使用 Gymnasium 的 `MountainCarContinuous-v0`。动作是 `[-1, 1]` 之间的连续油门，
现有依赖可以直接运行：

```powershell
python .\连续控制\benchmark_baselines.py
```

命令会评估零油门、随机油门、全油门惯性和平滑惯性四种规则基线，同时生成 JSON、CSV 和
教程配图。播放单条策略：

```powershell
python .\连续控制\play_baseline.py smooth_momentum
```

原始数据保存在
[`docs/experiments/09-continuous-baselines.json`](docs/experiments/09-continuous-baselines.json) 和
[`docs/experiments/09-continuous-baselines.csv`](docs/experiments/09-continuous-baselines.csv)。

## 连续 PPO 实验

训练单个种子：

```powershell
python .\连续控制\train_continuous_ppo.py --seed 42
```

使用 `7、42、2026` 三个训练种子完成实验，并与第九篇规则基线比较：

```powershell
python .\连续控制\benchmark_continuous_ppo.py
```

每个种子默认训练 5 万步，最终在相同的 100 个初始状态上进行确定性评估。实验数据保存在
[`docs/experiments/10-continuous-ppo.json`](docs/experiments/10-continuous-ppo.json) 和
[`docs/experiments/10-continuous-ppo.csv`](docs/experiments/10-continuous-ppo.csv)。播放检查点：

```powershell
python .\连续控制\play_continuous_ppo.py `
  .\runs\continuous-ppo\seed-42\checkpoints\best.pt
```

## SAC 实验

训练单个 SAC 种子：

```powershell
python .\连续控制\train_sac.py --seed 42
```

使用三个训练种子和 100 个固定测试起点完成实验：

```powershell
python .\连续控制\benchmark_sac.py
```

实验数据保存在 [`docs/experiments/11-sac.json`](docs/experiments/11-sac.json) 和
[`docs/experiments/11-sac.csv`](docs/experiments/11-sac.csv)。播放检查点：

```powershell
python .\连续控制\play_sac.py .\runs\sac\seed-42\checkpoints\best.pt
```

## PPO 与 SAC 对照

读取第十、十一篇已经保存的原始报告并生成配对比较：

```powershell
python .\连续控制\compare_ppo_sac.py
```

对照数据保存在
[`docs/experiments/12-ppo-vs-sac.json`](docs/experiments/12-ppo-vs-sac.json) 和
[`docs/experiments/12-ppo-vs-sac.csv`](docs/experiments/12-ppo-vs-sac.csv)。

## 二维连续动作入门

LunarLander 需要 Box2D，安装 `requirements.txt` 会自动安装对应扩展。运行四种基线：

```powershell
python .\月球着陆\benchmark_lunar_baselines.py
```

播放规则控制器：

```powershell
python .\月球着陆\play_lunar_baseline.py heuristic --seed 42
```

原始结果保存在
[`docs/experiments/13-lunar-baselines.json`](docs/experiments/13-lunar-baselines.json) 和
[`docs/experiments/13-lunar-baselines.csv`](docs/experiments/13-lunar-baselines.csv)。

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
  inspect_decisions.py # 六种模型的固定状态决策检查与可视化
连续控制/
  mountain_car_baselines.py # 连续山地车规则基线与回合评估
  benchmark_baselines.py # 100 个配对起点的统一实验入口
  play_baseline.py # 播放指定规则策略
  visualize_baselines.py # 实验结果与轨迹配图
  continuous_ppo.py # 高斯策略、连续 GAE 与 PPO 更新
  train_continuous_ppo.py # 单种子训练、评估和检查点
  benchmark_continuous_ppo.py # 三种子实验与规则基线比较
  play_continuous_ppo.py # 播放连续 PPO 检查点
  visualize_continuous_ppo.py # 第十篇实验配图
  sac.py # 回放池、连续 Actor、双 Q 网络和 SAC 更新
  train_sac.py # 单种子 SAC 训练与固定起点评估
  benchmark_sac.py # 三种子实验与时间相关探索对照
  play_sac.py # 播放 SAC 检查点
  visualize_sac.py # 第十一篇实验配图
  compare_ppo_sac.py # 校验并比较 PPO/SAC 原始实验报告
  visualize_ppo_sac.py # 第十二篇对照配图
月球着陆/
  lunar_baselines.py # 二维动作规则与随机基线
  benchmark_lunar_baselines.py # 固定起点实验和报告导出
  play_lunar_baseline.py # 图形化播放基线
  visualize_lunar_baselines.py # 第十三篇实验配图
  lunar_agents.py # 二维 PPO/SAC Agent 与观测归一化
  train_lunar_ppo.py # PPO 迁移训练与增量诊断
  train_lunar_sac.py # SAC 迁移训练与增量诊断
  visualize_lunar_diagnostics.py # 第十四篇失败诊断配图
  benchmark_lunar_ppo_stability.py # 第十五篇三种子稳定性汇总
  visualize_lunar_ppo_stability.py # 第十五篇实验配图
  benchmark_lunar_sac_stability.py # 第十六篇 SAC 容量与目标熵消融
  visualize_lunar_sac_stability.py # 第十六篇实验配图
  compare_lunar_agents.py # 第十七篇共同预算与固定场景比较
  visualize_lunar_comparison.py # 第十七篇实验配图
迷你网格/
  doorkey_env.py # DoorKey 环境、专家搜索与统一评估
  doorkey_dqn.py # 完整/局部观测 DQN
  doorkey_ppo.py # 前馈 PPO 与 GRU PPO
  train_doorkey_ppo.py # 课程学习、计数奖励与保守微调
  train_doorkey_imitation.py # 专家行为克隆
  benchmark_doorkey_generalization.py # 跨尺寸泛化评估
tutorials/        # 各篇教程的独立代码快照与运行入口
tests/            # 环境、模型、检查点和短训练测试
docs/csdn/        # 可发布到 CSDN 的后续教程
```

不熟悉 Git 的读者可以直接进入 [`tutorials/`](tutorials/README.md)，按文章编号找到对应代码，
不需要从提交历史中还原旧版本。根目录 [`贪吃蛇/`](贪吃蛇/) 始终保留项目的最新完整实现。

## 测试

```powershell
python -m pip install -r requirements-dev.txt
python -m ruff check .
python -m ruff format --check .
python -m pytest
```

## 教程

- [第一篇：速通贪吃蛇游戏](https://blog.csdn.net/bobwww123/article/details/138722671) · [对应代码](tutorials/01-snake-game/)
- [第二篇：手撕 GPT（零基础保姆级教学）](https://blog.csdn.net/bobwww123/article/details/138948884) · [对应代码](tutorials/02-actor-critic/)
- [第三篇：让贪吃蛇训练更稳定、更容易复现](https://blog.csdn.net/bobwww123/article/details/163925583) · [对应代码](tutorials/03-stable-training/)
- [第四篇：DQN 实战，四种策略同场对照](https://blog.csdn.net/bobwww123/article/details/163925932) · [对应代码](tutorials/04-dqn/)
- [第五篇：PPO 实战，从单步更新到稳定策略优化](https://blog.csdn.net/bobwww123/article/details/163926240) · [对应代码](tutorials/05-ppo/)
- [第六篇：Double DQN 实战，降低 Q 值成绩就会更好吗](https://blog.csdn.net/bobwww123/article/details/163937191) · [对应代码](tutorials/06-double-dqn/)
- [第七篇：Dueling DQN 实战，先判断局面，再选择动作](https://blog.csdn.net/bobwww123/article/details/163937626) · [对应代码](tutorials/07-dueling-dqn/)
- [第八篇：同一局面，六种模型会怎么走？](https://blog.csdn.net/bobwww123/article/details/164296013) · [对应代码](tutorials/08-decision-inspection/)
- [第九篇：油门不是开关，第一次走进连续动作空间](https://blog.csdn.net/bobwww123/article/details/164314437) · [对应代码](tutorials/09-continuous-action-basics/)
- [第十篇：油门到底该踩多少？用 PPO 学会连续控制](https://blog.csdn.net/bobwww123/article/details/164314648) · [对应代码](tutorials/10-continuous-ppo/)
- 第十一篇：经验也能反复学习，第一次实现 SAC · [文章草稿](docs/csdn/11-经验也能反复学习第一次实现SAC.md) · [对应代码](tutorials/11-sac/)
- 第十二篇：PPO 还是 SAC？同样 5 万步结果为什么不同 · [文章草稿](docs/csdn/12-PPO还是SAC同样5万步结果为什么不同.md) · [对应代码](tutorials/12-ppo-vs-sac/)
- 第十三篇：一根油门不够了，走进二维连续动作 · [文章草稿](docs/csdn/13-一根油门不够了走进二维连续动作.md) · [对应代码](tutorials/13-lunar-baselines/)
- 第十四篇：代码能运行，为什么月球还是落不下来 · [文章草稿](docs/csdn/14-代码能运行为什么月球还是落不下来.md) · [对应代码](tutorials/14-lunar-transfer-diagnostics/)
- 第十五篇：一次成功不算稳定，用三随机种子检验 PPO · [文章草稿](docs/csdn/15-一次成功不算稳定用三随机种子检验PPO.md) · [对应代码](tutorials/15-stable-lunar-ppo/)
- 第十六篇：网络更大、探索更多，SAC 就一定会着陆吗 · [文章草稿](docs/csdn/16-网络更大探索更多SAC就一定会着陆吗.md) · [对应代码](tutorials/16-lunar-sac-stability/)
- 第十七篇：PPO 还是 SAC？同预算与固定危险场景复盘 · [文章草稿](docs/csdn/17-PPO还是SAC同预算与固定危险场景复盘.md) · [对应代码](tutorials/17-lunar-ppo-vs-sac/)
- 第十八篇：门就在眼前，为什么没有钥匙还是过不去 · [文章草稿](docs/csdn/18-门就在眼前为什么没有钥匙还是过不去.md) · [对应代码](tutorials/18-doorkey-baselines/)
- 第十九篇：看见完整地图，DQN 就能稳定学会开门吗 · [文章草稿](docs/csdn/19-看见完整地图DQN就能稳定学会开门吗.md) · [对应代码](tutorials/19-full-observation-dqn/)
- 第二十篇：视野缩小以后，同一画面可能需要不同动作 · [文章草稿](docs/csdn/20-视野缩小以后同一画面可能需要不同动作.md) · [对应代码](tutorials/20-partial-observation/)
- 第二十一篇：记忆不是自动学会的：GRU 与 PPO 对照 · [文章草稿](docs/csdn/21-记忆不是自动学会的GRU与PPO对照.md) · [对应代码](tutorials/21-memory-ppo/)
- 第二十二篇：探索范围不等于任务进展：课程学习解决稀疏奖励 · [文章草稿](docs/csdn/22-探索范围不等于任务进展课程学习解决稀疏奖励.md) · [对应代码](tutorials/22-curriculum-exploration/)
- 第二十三篇：示范是起点，不是免疫牌：行为克隆与保守微调 · [文章草稿](docs/csdn/23-示范是起点不是免疫牌行为克隆与保守微调.md) · [对应代码](tutorials/23-imitation-ppo/)
- 第二十四篇：训练成功不等于泛化成功：DoorKey 跨尺寸评估 · [文章草稿](docs/csdn/24-训练成功不等于泛化成功DoorKey跨尺寸评估.md) · [对应代码](tutorials/24-generalization/)

[查看教程与代码的完整索引](tutorials/README.md)

## 后续路线

MiniGrid DoorKey 项目在第二十四篇完成了从环境基线、完整地图、局部观测、GRU、课程学习、
行为克隆到跨尺寸泛化的闭环。下一阶段可切换到真正 RGB 图像、多智能体协作或真实设备约束，
不再继续围绕 DoorKey 堆叠调参文章。

## License

MIT
