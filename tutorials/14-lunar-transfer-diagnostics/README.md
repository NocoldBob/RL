# 第十四篇：代码能运行，为什么月球还是落不下来

这一篇把 MountainCar 的 PPO/SAC 迁移到 LunarLander，并保留直接迁移失败、网络拆分后短暂
改善、最佳策略随后崩塌等真实结果。重点不是给出一组“成功参数”，而是建立下一轮稳定化所需
的诊断基线。

重新生成诊断配图：

```powershell
python .\tutorials\14-lunar-transfer-diagnostics\visualize_lunar_diagnostics.py
```

训练入口已经具备增量历史和最佳检查点保存：

```powershell
python .\tutorials\14-lunar-transfer-diagnostics\train_lunar_ppo.py --seed 42
python .\tutorials\14-lunar-transfer-diagnostics\train_lunar_sac.py --seed 42
```

原始诊断数据位于 `docs/experiments/14-lunar-transfer-diagnostics.json`。下一篇将在当前独立
Actor/Critic 基础上加入目标 KL 早停、降低更新轮数和更完整的策略稳定性指标。
