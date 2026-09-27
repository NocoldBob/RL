# 第 38 篇：可学习通信与消息消融

```powershell
python .\tutorials\38-learned-communication\train_marl_ppo.py `
  --mode communication --message-dropout 0.1 --total-steps 30000
python .\tutorials\38-learned-communication\benchmark_communication.py --episodes 50
```

测试时分别保留、随机丢弃和完全切断消息，判断 Actor 是否真正使用通信。
