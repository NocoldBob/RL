# 第七篇：Dueling DQN 实战

- [阅读 CSDN 正式文章](https://blog.csdn.net/bobwww123/article/details/163937626)
- 重点文件：`dqn.py`、`train_dqn.py`、`benchmark_dueling_dqn.py`

这一篇把 Q 网络拆成状态价值与动作优势，并用 `2×2` 实验比较普通/Dueling 网络结构和
普通/Double TD 目标。

训练 Dueling Double DQN：

```powershell
python .\tutorials\07-dueling-dqn\train_dqn.py --dueling --double-dqn `
  --output-dir runs\tutorial-07-dueling-double-dqn
```

运行四组配对实验：

```powershell
python .\tutorials\07-dueling-dqn\benchmark_dueling_dqn.py --seeds 7 42 2026 `
  --episodes 1000 --eval-episodes 100 --device cpu --torch-threads 1
```
