# 第 29 篇：驾驶奖励设计

比较环境原始奖励和加入转向、刹车、越界轻量惩罚的训练奖励。两个版本最终都按照环境原始回报评价。

```powershell
python .\tutorials\29-driving-reward-design\train_carracing_ppo.py `
  --total-steps 10000 --reward-mode shaped `
  --output-dir runs\carracing-ppo\shaped-seed-42
```

塑形项只影响训练信号，不改变最终成功阈值和模型排名口径。
