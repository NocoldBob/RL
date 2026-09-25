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
| 08 | 同一局面，六种模型会怎么走？ | [固定局面决策检查](https://blog.csdn.net/bobwww123/article/details/164296013) | [`08-decision-inspection/`](08-decision-inspection/) |
| 09 | 油门不是开关：连续动作入门 | [CSDN 正式文章](https://blog.csdn.net/bobwww123/article/details/164314437) | [`09-continuous-action-basics/`](09-continuous-action-basics/) |
| 10 | 油门到底该踩多少？连续 PPO | [CSDN 正式文章](https://blog.csdn.net/bobwww123/article/details/164314648) | [`10-continuous-ppo/`](10-continuous-ppo/) |
| 11 | 经验也能反复学习：第一次实现 SAC | [文章草稿](../docs/csdn/11-经验也能反复学习第一次实现SAC.md) | [`11-sac/`](11-sac/) |
| 12 | PPO 还是 SAC：同样 5 万步结果为什么不同 | [文章草稿](../docs/csdn/12-PPO还是SAC同样5万步结果为什么不同.md) | [`12-ppo-vs-sac/`](12-ppo-vs-sac/) |
| 13 | 一根油门不够了：走进二维连续动作 | [文章草稿](../docs/csdn/13-一根油门不够了走进二维连续动作.md) | [`13-lunar-baselines/`](13-lunar-baselines/) |
| 14 | 代码能运行，为什么月球还是落不下来 | [文章草稿](../docs/csdn/14-代码能运行为什么月球还是落不下来.md) | [`14-lunar-transfer-diagnostics/`](14-lunar-transfer-diagnostics/) |
| 15 | 一次成功不算稳定：三随机种子检验 PPO | [文章草稿](../docs/csdn/15-一次成功不算稳定用三随机种子检验PPO.md) | [`15-stable-lunar-ppo/`](15-stable-lunar-ppo/) |
| 16 | 网络更大、探索更多，SAC 就一定会着陆吗 | [文章草稿](../docs/csdn/16-网络更大探索更多SAC就一定会着陆吗.md) | [`16-lunar-sac-stability/`](16-lunar-sac-stability/) |
| 17 | PPO 还是 SAC：同预算与固定危险场景复盘 | [文章草稿](../docs/csdn/17-PPO还是SAC同预算与固定危险场景复盘.md) | [`17-lunar-ppo-vs-sac/`](17-lunar-ppo-vs-sac/) |

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
- 第 9 篇开启连续控制阶段，先建立环境、规则基线和统一指标，不提前引入新的训练算法。
- 第 10 篇把 PPO 扩展到连续动作，并保留多随机种子下成功与失败的真实结果。
- 第 11 篇加入 SAC，并通过时间相关随机探索解决惯性任务难以发现稀疏奖励的问题。
- 第 12 篇读取两份原始报告，在相同环境步数、训练种子和测试起点下比较 PPO 与 SAC。
- 第 13 篇切换到 LunarLander，用环境语义与四种基线开启二维连续控制阶段。
- 第 14 篇保留 PPO/SAC 直接迁移失败和策略崩塌，建立后续稳定化诊断基线。
- 第 15 篇加入价值裁剪、目标 KL 和完整更新诊断，并用三随机种子揭示剩余敏感性。
- 第 16 篇扩大 SAC 网络并检查目标熵与双 Q 分歧，保留“回报改善但未着陆”的负结果。
- 第 17 篇区分共同预算与最佳部署口径，并用固定危险状态解释 PPO/SAC 的行为差异。
- [`../贪吃蛇/`](../贪吃蛇/) 和 [`../连续控制/`](../连续控制/) 保存各阶段持续更新的实现；
  教程快照只为配合文章阅读，不会随以后章节的功能继续变化。
