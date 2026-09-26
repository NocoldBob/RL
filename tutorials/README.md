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
| 18 | DoorKey 环境与规则专家基线 | [文章草稿](../docs/csdn/18-门就在眼前为什么没有钥匙还是过不去.md) | [`18-doorkey-baselines/`](18-doorkey-baselines/) |
| 19 | 完整地图 DQN 与稀疏奖励 | [文章草稿](../docs/csdn/19-看见完整地图DQN就能稳定学会开门吗.md) | [`19-full-observation-dqn/`](19-full-observation-dqn/) |
| 20 | 局部观测与观测别名 | [文章草稿](../docs/csdn/20-视野缩小以后同一画面可能需要不同动作.md) | [`20-partial-observation/`](20-partial-observation/) |
| 21 | GRU 记忆与截断 BPTT | [文章草稿](../docs/csdn/21-记忆不是自动学会的GRU与PPO对照.md) | [`21-memory-ppo/`](21-memory-ppo/) |
| 22 | 访问计数与课程学习 | [文章草稿](../docs/csdn/22-探索范围不等于任务进展课程学习解决稀疏奖励.md) | [`22-curriculum-exploration/`](22-curriculum-exploration/) |
| 23 | 行为克隆与参考 KL 微调 | [文章草稿](../docs/csdn/23-示范是起点不是免疫牌行为克隆与保守微调.md) | [`23-imitation-ppo/`](23-imitation-ppo/) |
| 24 | DoorKey 跨尺寸泛化 | [文章草稿](../docs/csdn/24-训练成功不等于泛化成功DoorKey跨尺寸评估.md) | [`24-generalization/`](24-generalization/) |
| 25 | CarRacing RGB 输入管线与基线 | [文章草稿](../docs/csdn/25-从状态向量到RGB图像CarRacing视觉输入管线.md) | [`25-visual-input-pipeline/`](25-visual-input-pipeline/) |
| 26 | 单帧与 4 帧堆叠 | [文章草稿](../docs/csdn/26-一张图够不够判断车辆状态单帧与帧堆叠.md) | [`26-frame-stacking/`](26-frame-stacking/) |
| 27 | 离散动作视觉 Double DQN | [文章草稿](../docs/csdn/27-离散化方向盘后DQN能学会开车吗.md) | [`27-discrete-dqn-driving/`](27-discrete-dqn-driving/) |
| 28 | 连续动作视觉 PPO | [文章草稿](../docs/csdn/28-连续动作PPO如何同时控制转向油门和刹车.md) | [`28-continuous-ppo-driving/`](28-continuous-ppo-driving/) |
| 29 | 驾驶奖励设计 | [文章草稿](../docs/csdn/29-奖励越精细驾驶就一定越好吗.md) | [`29-driving-reward-design/`](29-driving-reward-design/) |
| 30 | 视觉扰动与数据增强 | [文章草稿](../docs/csdn/30-画面变暗以后策略还认识赛道吗.md) | [`30-visual-robustness/`](30-visual-robustness/) |
| 31 | 未见赛道与可靠性审计 | [文章草稿](../docs/csdn/31-训练成功不等于可靠驾驶视觉控制最终复盘.md) | [`31-driving-reliability/`](31-driving-reliability/) |

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
- 第 18 篇建立 DoorKey 环境、搜索专家和随机基线。
- 第 19 篇先用完整地图 DQN 分离规划与感知难点。
- 第 20 篇切换局部观测，统计同一画面对应多个动作的观测别名。
- 第 21 篇加入 GRU 与截断 BPTT，验证记忆结构不会自动产生记忆行为。
- 第 22 篇比较访问计数和课程学习，展示课程学习如何解决 8×8 稀疏奖励。
- 第 23 篇从专家轨迹行为克隆，再用参考策略 KL 抑制 PPO 灾难性遗忘。
- 第 24 篇在未见地图尺寸和随机种子上评估泛化能力。
- 第 25 篇建立 RGB 裁剪、缩放、帧跳过、动作映射和非学习基线。
- 第 26 篇在相同 Double DQN 预算下比较单帧与 4 帧时间表示。
- 第 27 篇拆解九动作视觉 Double DQN 的能力和动作精度上限。
- 第 28 篇改用三维连续高斯策略与 CNN-PPO。
- 第 29 篇比较原始奖励和行为塑形，并统一按环境原始回报评价。
- 第 30 篇加入训练时图像增强，并在五类受控像素扰动下测试。
- 第 31 篇使用未见赛道、尾部回报、越界率、动作抖动和延迟统一复盘。
- [`../贪吃蛇/`](../贪吃蛇/) 和 [`../连续控制/`](../连续控制/) 保存各阶段持续更新的实现；
  教程快照只为配合文章阅读，不会随以后章节的功能继续变化。
