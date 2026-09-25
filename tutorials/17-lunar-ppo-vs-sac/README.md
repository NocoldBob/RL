# 第十七篇：PPO 还是 SAC？

本快照读取第 15、16 篇的原始训练日志与最佳检查点，生成两种对照：

- 相同 `seed=42`、10 万步和 20 个测试起点的学习效率比较；
- 各自最佳检查点在 100 个新起点和五个固定危险状态上的部署复盘。

从仓库根目录运行：

```powershell
python tutorials/17-lunar-ppo-vs-sac/compare_lunar_agents.py
python tutorials/17-lunar-ppo-vs-sac/visualize_lunar_comparison.py
```

脚本默认读取仓库根目录的 `runs/lunar-ppo-stable-v2/` 和 `runs/lunar-sac-stable-128/`。

