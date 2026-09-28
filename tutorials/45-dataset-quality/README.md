# 第 45 篇：数据质量消融

在相同 CQL 配置下分别训练 random、medium、expert 和 mixed 数据，再运行：

```powershell
python .\tutorials\45-dataset-quality\benchmark_offline_agents.py
```

真实结果表明 transition 更多并不等于策略更好。
