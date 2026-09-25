# 第十三篇：一根油门不够了，走进二维连续动作

这一篇从连续山地车切换到 `LunarLanderContinuous-v3`。环境提供 8 维状态和 2 维连续动作，
分别控制主发动机与侧向发动机。

先安装包含 Box2D 的项目依赖：

```powershell
python -m pip install -r requirements.txt
```

运行四种基线的 100 个固定起点实验：

```powershell
python .\tutorials\13-lunar-baselines\benchmark_lunar_baselines.py
```

播放规则控制器：

```powershell
python .\tutorials\13-lunar-baselines\play_lunar_baseline.py heuristic --seed 42
```

输出保存在：

- `docs/experiments/13-lunar-baselines.json`
- `docs/experiments/13-lunar-baselines.csv`
- `docs/assets/csdn-13/`

本篇只建立环境语义、指标和规则上下界，不训练神经网络。下一篇将在相同测试起点上迁移 PPO
与 SAC。
