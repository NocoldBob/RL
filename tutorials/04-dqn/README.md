# 第四篇：DQN 实战

- [阅读 CSDN 正式文章](https://blog.csdn.net/bobwww123/article/details/163925932)
- 重点文件：`dqn.py`、`train_dqn.py`、`benchmark.py`

这一篇加入经验回放、目标网络和 ε-greedy 探索，并统一比较随机策略、教师策略、单步
Actor-Critic 与 DQN。

```powershell
python .\tutorials\04-dqn\train_dqn.py --output-dir runs\tutorial-04-dqn
```

```powershell
python .\tutorials\04-dqn\benchmark.py --seeds 7 42 2026 --episodes 1000 `
  --eval-episodes 100 --device cpu --torch-threads 1
```
