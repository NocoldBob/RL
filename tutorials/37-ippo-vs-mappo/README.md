# 第 37 篇：IPPO 与 MAPPO

```powershell
python .\tutorials\37-ippo-vs-mappo\train_marl_ppo.py --mode ippo --total-steps 30000
python .\tutorials\37-ippo-vs-mappo\train_marl_ppo.py --mode mappo --total-steps 30000
```

两个版本共享 Actor，只改变 Critic 使用局部观测还是训练期全局状态。
