# 第一篇：速通贪吃蛇游戏

- [阅读 CSDN 正式文章](https://blog.csdn.net/bobwww123/article/details/138722671)
- 重点文件：`environment.py`

这一篇先理解地图、蛇身、食物、相对动作、奖励和终止条件。目录同时保留后续训练入口，方便
读者完成环境部分后直接观察智能体怎样调用它。

短流程检查（请在仓库根目录运行）：

```powershell
python .\tutorials\01-snake-game\main.py --episodes 20 --teacher-episodes 5 `
  --grid-size 6 --end-score 4 --max-steps 100 --eval-interval 10 --eval-episodes 3 `
  --output-dir runs\tutorial-01
```
