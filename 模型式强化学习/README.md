# 模型式强化学习：Pendulum-v1

第 49–57 篇从动力学数据开始，依次实现多步误差诊断、Random Shooting MPC、CEM、模型集成、Dyna、简化 MBPO、离线模型规划和可靠性审计。

```powershell
python .\模型式强化学习\collect_dynamics_data.py
python .\模型式强化学习\train_dynamics.py
python .\模型式强化学习\rollout_diagnostics.py
python .\模型式强化学习\benchmark_planning.py
python .\模型式强化学习\benchmark_ensemble.py
python .\模型式强化学习\benchmark_imagination.py
python .\模型式强化学习\benchmark_offline_planning.py
python .\模型式强化学习\benchmark_model_reliability.py
python .\模型式强化学习\visualize_model_based_rl.py
```

真实数据和实验 JSON 提交到仓库；动力学与策略 checkpoint 写入被 Git 忽略的 `runs/`。所有模型选择与最终扰动审计使用不同种子。

主要结果：学习模型 MPC 基本追平真实模型 MPC；CEM 明显超过均匀随机搜索；模型集成和不确定性惩罚在干净条件没有自动改善；短 MBPO rollout 之间差异很小；最终审计中 MBPO 的整体回报最好。
