# 第十二篇：PPO 还是 SAC，同样 5 万步结果为什么不同

这一篇不重新训练，也不重新调参。比较程序直接读取第十篇和第十一篇保存的原始实验报告，
先校验训练种子、环境步数和测试起点完全一致，再生成逐种子比较、汇总数据和文章配图。

从仓库根目录运行：

```powershell
python .\tutorials\12-ppo-vs-sac\compare_ppo_sac.py
```

输入数据：

- `docs/experiments/10-continuous-ppo.json`
- `docs/experiments/11-sac.json`

输出数据：

- `docs/experiments/12-ppo-vs-sac.json`
- `docs/experiments/12-ppo-vs-sac.csv`
- `docs/assets/csdn-12/`

比较结论仅适用于当前实现和预算。两种算法的环境交互步数相同，但 SAC 的网络数量和梯度更新
工作量明显更大，因此“样本效率相同”不等于“计算成本相同”。
