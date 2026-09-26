# 第 31 篇：视觉驾驶可靠性复盘

在未参与训练的赛道种子上统一评估离散 DQN、原始 PPO、塑形 PPO 和增强 PPO，并报告均值、中位数、10% 分位、最差回报、越界率、动作抖动与推理延迟。

```powershell
python .\tutorials\31-driving-reliability\benchmark_visual_reliability.py --episodes 10
```

最终结论以回报分布和尾部风险为主，不以某次最高分代表部署可靠性。
