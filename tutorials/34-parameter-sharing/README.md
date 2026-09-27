# 第 34 篇：参数共享与 Agent ID

```powershell
python .\tutorials\34-parameter-sharing\train_value_agents.py --mode shared --total-steps 20000
python .\tutorials\34-parameter-sharing\train_value_agents.py --mode shared --agent-id --total-steps 20000
```

比较独立网络、纯共享网络和追加 one-hot 身份的共享网络。
