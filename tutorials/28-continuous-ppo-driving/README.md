# 第 28 篇：连续动作视觉 PPO

CNN-PPO 同时输出转向、油门和刹车，通过 `tanh` 压缩与动作范围映射控制 `CarRacing-v3`。

```powershell
python .\tutorials\28-continuous-ppo-driving\train_carracing_ppo.py `
  --total-steps 10000 --reward-mode raw `
  --output-dir runs\carracing-ppo\raw-seed-42
```

评估仍使用环境原始回报，并与第 27 篇的离散 Double DQN 进行相同赛道种子对照。
