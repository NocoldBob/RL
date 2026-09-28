# 第 42 篇：离线 DQN 与外推误差

```powershell
python .\tutorials\42-offline-dqn\train_offline.py --mode dqn --dataset datasets\acrobot\mixed.npz --output-dir runs\offline\dqn-mixed
```

展示固定 replay 上的 `max Q` 如何选择缺少数据支持的动作。
