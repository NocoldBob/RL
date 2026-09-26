# 第 26 篇：单帧与帧堆叠

保持 Double DQN、离散动作、训练预算和评估种子相同，只比较单帧与 4 帧输入。

```powershell
python .\tutorials\26-frame-stacking\train_carracing_dqn.py `
  --frame-stack 1 --total-steps 10000 `
  --output-dir runs\carracing-dqn\stack1-seed-42
python .\tutorials\26-frame-stacking\train_carracing_dqn.py `
  --frame-stack 4 --total-steps 10000 `
  --output-dir runs\carracing-dqn\stack4-seed-42
```

图像中的位置可以由单帧观察，速度和运动方向则需要时间信息。帧堆叠也会增加卷积网络输入和回放缓存体积。
