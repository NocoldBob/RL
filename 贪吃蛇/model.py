import torch
import torch.nn as nn


class ConvActorCritic(nn.Module):
    def __init__(self, input_channels, output_dim, grid_size, lr=1e-3, weight_decay=1e-5):
        super(ConvActorCritic, self).__init__()
        self.conv1 = 8
        self.conv2 = 16
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_channels, self.conv1, kernel_size=3, stride=1, padding=1),
            # nn.MaxPool2d(2, 2),  # 下采样使用一次，保证grid_size能被整除
            nn.ReLU(),
            # nn.Conv2d(self.conv1, self.conv2, kernel_size=3, stride=1, padding=1),
            # nn.MaxPool2d(2, 2),  # 下采样
            # nn.ReLU(),
            nn.Flatten()
        )

        reduced_grid_size = grid_size // 1  # 下采样后，特征边缘缩小2倍
        self.actor = nn.Sequential(
            nn.Linear(self.conv1 * reduced_grid_size * reduced_grid_size, output_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Linear(self.conv1 * reduced_grid_size * reduced_grid_size, 1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr, weight_decay=weight_decay)

    def forward(self, x):
        features = self.feature_extractor(x)
        action_probs = self.actor(features)
        value = self.critic(features).squeeze(-1)
        return action_probs, value

    def update(self, state, action, reward, next_state, done, gamma=0.99):
        current_probs, current_value = self(state)
        _, next_value = self(next_state)

        td_target = reward + (1 - int(done)) * gamma * next_value
        td_error = td_target - current_value

        dist = torch.distributions.Categorical(current_probs)
        log_prob = dist.log_prob(action)
        actor_loss = -(log_prob * td_error.detach()).mean()
        critic_loss = 0.5 * td_error.pow(2).mean()
        total_loss = actor_loss + critic_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

    def save_model(self, model, filename="model.pth"):
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, filename)

    def load_model(self, model, filename="model.pth"):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
