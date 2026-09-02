# 强化学习教程配套代码

这里按篇章编号保存可以直接查看和运行的代码快照。初学者不需要使用 `git checkout`，也不需要
在提交历史中猜测某篇文章对应哪个版本。

| 篇章 | 主题 | 文章 | 代码 |
|---|---|---|---|
| 01 | 贪吃蛇环境与游戏规则 | [速通贪吃蛇游戏](https://blog.csdn.net/bobwww123/article/details/138722671) | [`01-snake-game/`](01-snake-game/) |
| 02 | 单步 Actor-Critic | [手撕 GPT](https://blog.csdn.net/bobwww123/article/details/138948884) | [`02-actor-critic/`](02-actor-critic/) |
| 03 | 可复现训练与独立评估 | [让训练更稳定、更容易复现](https://blog.csdn.net/bobwww123/article/details/163925583) | [`03-stable-training/`](03-stable-training/) |
| 04 | DQN 与四策略对照 | [DQN 实战](https://blog.csdn.net/bobwww123/article/details/163925932) | [`04-dqn/`](04-dqn/) |
| 05 | PPO、Rollout 与 GAE | [PPO 实战](https://blog.csdn.net/bobwww123/article/details/163926240) | [`05-ppo/`](05-ppo/) |
| 06 | Double DQN 与 Q 值诊断 | [Double DQN 实战](https://blog.csdn.net/bobwww123/article/details/163937191) | [`06-double-dqn/`](06-double-dqn/) |
| 07 | Dueling Network 全因子实验 | [Dueling DQN 实战](https://blog.csdn.net/bobwww123/article/details/163937626) | [`07-dueling-dqn/`](07-dueling-dqn/) |
| 08 | 同一局面，六种模型会怎么走？ | 正式文章发布后补充 | [`08-decision-inspection/`](08-decision-inspection/) |

## 使用方法

先在仓库根目录安装一次依赖：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

然后进入对应篇章，按照该目录中的 `README.md` 运行。训练结果统一写入仓库根目录的 `runs/`，
不会提交到 GitHub。

## 快照说明

- 第 1 至第 3 篇采用修正后的 Gymnasium 与 Actor-Critic 教学基线，避免新读者被旧依赖和早期
  实现细节卡住；三篇关注点不同，因此分别保留入口。
- 第 4 至第 7 篇保留该算法加入课程时的代码范围，便于逐步观察 DQN、PPO、Double DQN 和
  Dueling DQN 分别增加了什么；第 8 篇保留完整离散控制实现，用于统一检查六种模型的决策。
- [`../贪吃蛇/`](../贪吃蛇/) 是持续更新的最新实现；教程快照只为配合文章阅读，不会随以后
  章节的功能继续变化。
