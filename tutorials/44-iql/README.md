# 第 44 篇：Implicit Q-Learning

```powershell
python .\tutorials\44-iql\train_offline.py --mode iql --dataset datasets\acrobot\mixed.npz --output-dir runs\offline\iql-mixed
```

使用 expectile 价值回归与优势加权行为克隆，避免对数据外动作取最大值。
