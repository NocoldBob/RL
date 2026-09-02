# 第九篇：油门不是开关，第一次走进连续动作空间

- [CSDN 正式文章](https://blog.csdn.net/bobwww123/article/details/164314437)
- 环境：`MountainCarContinuous-v0`
- 重点文件：`mountain_car_baselines.py`、`benchmark_baselines.py`

这一篇从贪吃蛇的三个离散动作切换到 `[-1, 1]` 连续油门。代码比较零油门、随机油门、
全油门惯性和平滑惯性四种无需训练的基线，并记录奖励、成功率、步数和动作成本。

## 运行完整实验

在仓库根目录执行：

```powershell
python .\tutorials\09-continuous-action-basics\benchmark_baselines.py
```

默认使用 100 个配对起点，生成：

- `docs/experiments/09-continuous-baselines.json`
- `docs/experiments/09-continuous-baselines.csv`
- `docs/assets/csdn-09/` 下的五张配图

## 播放策略

```powershell
python .\tutorials\09-continuous-action-basics\play_baseline.py smooth_momentum
```

可选策略为 `zero`、`random`、`bang_bang` 和 `smooth_momentum`。这些策略都是规则基线，不是
训练完成的强化学习模型。
