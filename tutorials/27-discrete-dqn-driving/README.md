# 第 27 篇：离散动作视觉 Double DQN

本篇将转向、油门和刹车压缩为 9 个组合动作，用卷积编码器、经验回放和 Double DQN 目标训练视觉驾驶策略。

```powershell
python .\tutorials\27-discrete-dqn-driving\train_carracing_dqn.py `
  --frame-stack 4 --total-steps 10000 `
  --output-dir runs\carracing-dqn\stack4-seed-42
```

训练记录包含独立评估结果；文章中的曲线不使用训练局回报冒充测试成绩。
