# 第 35 篇：VDN 与信用分配

```powershell
python .\tutorials\35-vdn-credit-assignment\train_value_agents.py `
  --mode vdn --agent-id --reward-mode shared --total-steps 20000
python .\tutorials\35-vdn-credit-assignment\train_value_agents.py `
  --mode vdn --agent-id --reward-mode assignment_progress --total-steps 20000
```

比较纯团队奖励和追加目标匹配进展的差分式训练信号。
