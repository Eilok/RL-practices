import torch.nn as nn
import torch
import torch.nn.functional as F

class Discriminator(nn.Module):
    ''' 判断给定动作状态对有多大可能来自于生成器 '''
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))
