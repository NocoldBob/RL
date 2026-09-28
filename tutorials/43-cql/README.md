# 第 43 篇：Conservative Q-Learning

```powershell
python .\tutorials\43-cql\train_offline.py --mode cql --cql-alpha 1 --dataset datasets\acrobot\mixed.npz --output-dir runs\offline\cql-mixed-alpha-1
```

在 Bellman loss 上加入保守惩罚，并比较 `alpha=0.1/1/5`。
