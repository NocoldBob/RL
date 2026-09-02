# 第三篇：让训练更稳定、更容易复现

- [阅读 CSDN 正式文章](https://blog.csdn.net/bobwww123/article/details/163925583)
- 重点文件：`environment.py`、`model.py`、`main.py`、`play.py`

这一篇关注显式教师阶段、方向观测、训练与独立评估隔离、随机种子、检查点和 TensorBoard。

```powershell
python .\tutorials\03-stable-training\main.py --output-dir runs\tutorial-03
```

```powershell
tensorboard --logdir .\runs\tutorial-03\tensorboard
```
