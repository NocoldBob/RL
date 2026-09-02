# 第八篇：同一局面，六种模型会怎么走？

- [阅读 CSDN 正式文章](https://blog.csdn.net/bobwww123/article/details/164296013)
- 重点文件：`inspect_decisions.py`

这一篇不增加新算法，而是让六个已训练模型读取四个完全相同的固定局面，比较它们选择的
动作，并查看策略概率、Q 值以及 Dueling Network 的状态价值和动作优势。

## 准备检查点

在仓库根目录运行：

```powershell
python .\tutorials\08-decision-inspection\benchmark.py `
  --seeds 7 42 2026 --episodes 1000 --eval-episodes 100 `
  --device cpu --torch-threads 1 --output-dir runs\benchmark-ppo

python .\tutorials\08-decision-inspection\benchmark_dueling_dqn.py `
  --seeds 7 42 2026 --episodes 1000 --eval-episodes 100 `
  --device cpu --torch-threads 1
```

已有检查点时可增加 `--reuse`，只重新评估和汇总。

## 生成决策报告

```powershell
python .\tutorials\08-decision-inspection\inspect_decisions.py
```

默认输出：

- `docs/experiments/08-decision-inspection.json`：完整状态、动作安全性和模型输出；
- `docs/assets/csdn-08/`：固定局面、动作矩阵和 Dueling 分解图。

脚本只读取模型并执行前向推理，不会继续训练或改写检查点。固定局面检查用于解释具体决策，
不能替代多随机种子的完整游戏评估。
