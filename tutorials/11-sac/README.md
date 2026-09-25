# 第十一篇：经验也能反复学习，第一次实现 SAC

这一篇继续使用 `MountainCarContinuous-v0`，从连续 PPO 切换到离策略算法 SAC。实现包含经验
回放、双 Q 网络、目标网络软更新、带 `tanh` 修正的高斯 Actor 和自动温度参数。

## 运行

从仓库根目录训练一个种子：

```powershell
python .\tutorials\11-sac\train_sac.py --seed 42
```

运行三个训练种子、100 个固定测试起点的完整实验：

```powershell
python .\tutorials\11-sac\benchmark_sac.py
```

播放最佳检查点：

```powershell
python .\tutorials\11-sac\play_sac.py .\runs\sac\seed-42\checkpoints\best.pt
```

完整实验默认生成：

- `docs/experiments/11-sac.json`
- `docs/experiments/11-sac.csv`
- `docs/assets/csdn-11/`

## 文件说明

- `sac.py`：经验回放、Actor、双 Q 网络、温度参数和 SAC 更新；
- `train_sac.py`：单种子训练、固定起点评估和检查点；
- `benchmark_sac.py`：探索对照、三训练种子实验和报告导出；
- `play_sac.py`：确定性策略图形化播放；
- `visualize_sac.py`：生成文章配图；
- `mountain_car_baselines.py`：与第九篇保持一致的环境和规则基线。

训练前 5000 步采用随机油门，并将每个随机动作保持 10 步。这是时间相关随机探索，不使用
规则策略、专家示范或人工奖励塑形。逐步独立随机动作在该惯性任务中很难形成有效摆动。
