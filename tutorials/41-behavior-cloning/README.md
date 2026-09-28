# 第 41 篇：行为克隆

```powershell
python .\tutorials\41-behavior-cloning\train_offline.py --mode bc --dataset datasets\acrobot\expert.npz --output-dir runs\offline\bc-expert
```

用纯监督学习模仿固定数据，并比较 expert 与 mixed 数据的标签一致性。
