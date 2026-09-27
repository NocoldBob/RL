# 多智能体协作：第 32–39 篇共享实现

主环境为 MPE2 `simple_spread_v3`，默认使用 3 个同质智能体、3 个目标、5 个离散动作和 25 步回合。

```powershell
python .\多智能体\benchmark_spread_baselines.py --episodes 100
python .\多智能体\train_value_agents.py --mode iql --total-steps 20000
python .\多智能体\train_value_agents.py --mode qmix --agent-id --reward-mode shared
python .\多智能体\train_marl_ppo.py --mode mappo --total-steps 30000
python .\多智能体\benchmark_marl_reliability.py --episodes 30
```

统一记录团队回报、目标覆盖率、全覆盖率、目标距离、碰撞率和推理延迟。全局状态只允许进入训练期混合网络或集中 Critic，分散执行的 Actor/Q 网络只读取局部观测。
