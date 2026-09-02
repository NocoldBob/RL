# 第十篇：油门到底该踩多少？用 PPO 学会连续控制

[CSDN 正式文章](https://blog.csdn.net/bobwww123/article/details/164314648)

本目录是第十篇教程的代码快照。它把离散动作 PPO 改为 `tanh` 压缩的高斯策略，并在
`MountainCarContinuous-v0` 上与第九篇规则基线使用相同起点评估。

在仓库根目录安装依赖后，训练单个种子：

```powershell
python .\tutorials\10-continuous-ppo\train_continuous_ppo.py --seed 42
```

运行三个训练种子的完整实验：

```powershell
python .\tutorials\10-continuous-ppo\benchmark_continuous_ppo.py
```

播放最佳检查点：

```powershell
python .\tutorials\10-continuous-ppo\play_continuous_ppo.py `
  .\runs\continuous-ppo\seed-42\checkpoints\best.pt
```

默认完整实验需要在 CPU 上完成 15 万次环境交互。测试报告写入
`docs/experiments/10-continuous-ppo.json` 和 `.csv`，配图写入 `docs/assets/csdn-10/`。

主要文件：

- `continuous_ppo.py`：高斯 Actor-Critic、动作概率修正、GAE 与 PPO 更新；
- `train_continuous_ppo.py`：单种子训练、独立评估和检查点；
- `benchmark_continuous_ppo.py`：三种子训练与第九篇基线对比；
- `play_continuous_ppo.py`：图形化播放检查点；
- `visualize_continuous_ppo.py`：生成文章配图。
