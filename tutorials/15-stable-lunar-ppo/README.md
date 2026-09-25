# 第十五篇：一次成功不算稳定

本快照包含 LunarLander PPO 的稳定化实现、三随机种子汇总和绘图入口。

核心变化：

- Actor/Critic 使用独立编码器；
- 初始动作 `log_std=-0.5`；
- 10 轮 PPO 更新、裁剪价值损失和目标 KL 保护；
- 记录梯度范数、价值解释率、KL、裁剪比例与策略熵；
- 固定 20 个测试起点，禁止把单个最好种子当作最终结论。

从仓库根目录运行：

```powershell
python tutorials/15-stable-lunar-ppo/train_lunar_ppo.py --seed 42 `
  --total-steps 200000 --eval-interval 25000 --eval-episodes 20 `
  --output-dir runs/lunar-ppo-stable-v2/seed-42
```

文章对应的完整三种子数据已经保存到 `docs/experiments/15-lunar-ppo-stability.json`。
