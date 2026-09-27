# 第 33 篇：Independent DQN

```powershell
python .\tutorials\33-independent-dqn\train_value_agents.py `
  --mode iql --total-steps 20000 --output-dir runs\marl-value\iql
```

三个独立 Double DQN 把不断变化的队友当成环境，展示经验回放面对多智能体非平稳性的限制。
