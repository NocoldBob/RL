# 第 40 篇：构建版本化离线数据集

```powershell
python .\tutorials\40-offline-dataset\train_behavior_dqn.py
python .\tutorials\40-offline-dataset\collect_dataset.py
python .\tutorials\40-offline-dataset\compile_dataset_report.py
```

保存轨迹边界、终止语义、行为动作概率与数据来源，并生成 random、medium、expert、mixed 四份 Acrobot 数据。
