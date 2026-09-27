# 第 36 篇：QMIX 单调混合网络

```powershell
python .\tutorials\36-qmix\train_value_agents.py `
  --mode qmix --agent-id --reward-mode shared --total-steps 20000
```

混合器训练时读取填充后的全局状态，执行时局部 Q 网络仍只使用各自观测。
