# 第六篇：Double DQN 实战

- [阅读 CSDN 正式文章](https://blog.csdn.net/bobwww123/article/details/163937191)
- 重点文件：`dqn.py`、`train_dqn.py`、`benchmark_double_dqn.py`

这一篇在相同网络与预算下配对比较 DQN 和 Double DQN，并记录 Q 值诊断信息。

```powershell
python .\tutorials\06-double-dqn\train_dqn.py --double-dqn `
  --output-dir runs\tutorial-06-double-dqn
```

```powershell
python .\tutorials\06-double-dqn\benchmark_double_dqn.py --seeds 7 42 2026 `
  --episodes 1000 --eval-episodes 100 --device cpu --torch-threads 1
```
