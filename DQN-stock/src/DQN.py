###### 实现DQN算法 ######
from Qnet import Qnet

import torch
import numpy as np
import torch.nn.functional as F

class DQN:
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma, epsilon, target_update, device):
        self.action_dim = action_dim
        self.device = device

        # Q网络（训练）
        self.q_net = Qnet(state_dim, hidden_dim, self.action_dim).to(self.device)
        # 目标网络（评估）
        self.target_q_net = Qnet(state_dim, hidden_dim, self.action_dim).to(self.device)
        # adam优化器
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)

        # 折扣因子
        self.gamma = gamma
        # epsilon贪婪策略参数
        self.epsilon = epsilon
        # 目标网络更新频率
        self.target_update = target_update
        # 记录更新次数
        self.count = 0

    def take_action(self, state):
        ''' epsilon-贪婪策略 '''
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim) # 随机选择动作
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item() # 选出Q值最大的动作
        return action

    def update(self, transition_dict):
        ''' 更新Q网络 '''
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        # 选取动作对应的Q值
        q_values = self.q_net(states).gather(1, actions)
        # 选取下一个状态的最大Q值
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        # TD误差目标
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)

        # 均方误差
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict()) # 更新目标网络
        self.count += 1