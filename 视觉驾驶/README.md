# 视觉驾驶：CarRacing 第 25–31 篇共享实现

本目录保存视觉强化学习阶段持续更新的完整实现；`tutorials/25-*` 至 `tutorials/31-*` 保存配合文章阅读的独立快照。

统一实验约定：

- 原始观测：`96×96×3` RGB；
- 处理观测：去除底部 HUD，缩放到 `64×64`，默认灰度；
- 默认时间表示：4 帧堆叠；
- 默认帧跳过：每次策略决策执行 4 个环境物理步；
- 成功阈值：单局环境原始回报不低于 200；
- 训练奖励允许缩放或塑形，所有模型最终都用原始回报评价。

主要入口：

```powershell
python .\视觉驾驶\benchmark_visual_baselines.py --episodes 5
python .\视觉驾驶\train_carracing_dqn.py --frame-stack 4 --total-steps 10000
python .\视觉驾驶\train_carracing_ppo.py --total-steps 10000
python .\视觉驾驶\benchmark_visual_robustness.py --episodes 5
python .\视觉驾驶\benchmark_visual_reliability.py --episodes 10
```

训练检查点写入 `runs/`，机器可读实验结果写入 `docs/experiments/`，文章配图写入 `docs/assets/csdn-25` 至 `csdn-31`。
