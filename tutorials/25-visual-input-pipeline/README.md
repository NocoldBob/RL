# 第 25 篇：视觉输入管线与基线

本篇建立 `CarRacing-v3` 的统一图像预处理、帧跳过、动作映射和评价指标，并比较静止、直行、随机与特权轨迹跟随基线。

在仓库根目录运行：

```powershell
python .\tutorials\25-visual-input-pipeline\benchmark_visual_baselines.py --episodes 5
python .\tutorials\25-visual-input-pipeline\visualize_visual_driving.py `
  --lesson 25 --input docs\experiments\25-visual-baselines.json `
  --output-dir docs\assets\csdn-25
```

特权轨迹跟随器读取模拟器内部轨迹点，只用于验证任务和指标，不是可部署的视觉策略。
