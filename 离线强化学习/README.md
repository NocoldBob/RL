# 离线强化学习：Acrobot-v1

第 40–48 篇使用同一套 Acrobot 数据和严格分离的训练、模型选择、最终评估流程。

## 推荐顺序

```powershell
python .\离线强化学习\train_behavior_dqn.py
python .\离线强化学习\collect_dataset.py
python .\离线强化学习\train_offline.py --mode bc --dataset datasets\acrobot\expert.npz
python .\离线强化学习\benchmark_offline_agents.py
python .\离线强化学习\benchmark_off_policy_evaluation.py
python .\离线强化学习\finetune_offline.py
python .\离线强化学习\benchmark_offline_reliability.py
python .\离线强化学习\visualize_offline_rl.py
```

`train_offline.py` 只读取固定数据，不导入或调用环境。`benchmark_*` 在 checkpoint 完成选择后才使用隐藏种子评估。仓库提交四份版本化数据与实验 JSON；模型 checkpoint 写入被 Git 忽略的 `runs/`。

## 核心结论

- expert BC 为 100% 成功，mixed BC 为 40%；
- mixed 数据上的普通离线 DQN 与三档 CQL 均为 0%；
- mixed IQL 为 59%，expert CQL 为 99%；
- 5,000 步在线预算中 mixed replay 最终为 65%；
- OPE 存在明显偏差，必须与支持度和不确定性一起解释。
