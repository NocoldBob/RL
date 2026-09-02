# 第二篇：手撕 GPT

- [阅读 CSDN 正式文章](https://blog.csdn.net/bobwww123/article/details/138948884)
- 重点文件：`model.py`、`main.py`

这一篇沿用第一篇的环境，加入卷积 Actor、Critic 和单步 TD 更新。代码采用当前可运行的
Gymnasium 教学基线，文章中的核心学习路线不变。

```powershell
python .\tutorials\02-actor-critic\main.py --episodes 100 --teacher-episodes 20 `
  --output-dir runs\tutorial-02
```

播放训练结果：

```powershell
python .\tutorials\02-actor-critic\play.py `
  .\runs\tutorial-02\checkpoints\best.pt --fps 15
```
