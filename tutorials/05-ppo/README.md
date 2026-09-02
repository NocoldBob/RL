# 第五篇：PPO 实战

- [阅读 CSDN 正式文章](https://blog.csdn.net/bobwww123/article/details/163926240)
- 重点文件：`ppo.py`、`train_ppo.py`、`play_ppo.py`

这一篇加入 Rollout、GAE、PPO Clip 和批量多轮更新，并将 PPO 放入统一基准。

```powershell
python .\tutorials\05-ppo\train_ppo.py --output-dir runs\tutorial-05-ppo
```

```powershell
python .\tutorials\05-ppo\play_ppo.py `
  .\runs\tutorial-05-ppo\checkpoints\best.pt --fps 15
```
