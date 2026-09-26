# 第 30 篇：视觉扰动与数据增强

训练带亮度和小幅平移增强的 PPO，然后在亮度、噪声、偏色、模糊和遮挡条件下统一评估。

```powershell
python .\tutorials\30-visual-robustness\train_carracing_ppo.py `
  --total-steps 10000 --augmentation `
  --output-dir runs\carracing-ppo\augmented-seed-42
python .\tutorials\30-visual-robustness\benchmark_visual_robustness.py --episodes 5
```

增强只作用于 PPO 更新批次，环境轨迹仍由原始观测采集，方便分离“训练时不变性”和“测试时扰动”。
