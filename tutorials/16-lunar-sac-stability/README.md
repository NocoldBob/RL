# 第十六篇：网络更大、探索更多，SAC 就一定会着陆吗？

本快照包含 LunarLander SAC 的 128×128 Actor/双 Q 实现，以及目标熵消融和 Q 分歧诊断。

从仓库根目录运行：

```powershell
python tutorials/16-lunar-sac-stability/train_lunar_sac.py --seed 42 `
  --total-steps 100000 --eval-interval 25000 --eval-episodes 20 `
  --output-dir runs/lunar-sac-stable-128/seed-42
```

本篇的结论是负结果：回报改善，但 10 万步内没有安全着陆。代码和文章会完整保留这次结果。

