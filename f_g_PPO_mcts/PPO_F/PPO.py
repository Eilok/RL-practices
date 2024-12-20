from PPO_F.network import ConvPolicyNet, ConvValueNet
from utils import compute_advantage

import torch
import torch.nn.functional as F


class PPO:
    ''' PPO算法,采用截断方式 '''
    def __init__(self, input_channels, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device):
        self.actor = ConvPolicyNet(input_channels, action_dim).to(device)
        self.critic = ConvValueNet(input_channels).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs  # 一条序列的数据用来训练轮数
        self.eps = eps  # 截断范围的参数
        self.device = device

    def take_action(self, state, mask):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        # print(state.shape)
        probs = self.actor(state)
        # print(probs)
        legal_probs = probs * mask # 保留合法动作概率
        # 无法移动，则保持静止
        if legal_probs.sum() == 0:
            return 16
        # 可移动，则选择最大概率动作
        else:
            legal_probs /= legal_probs.sum()
            # print(legal_probs)
            action_dist = torch.distributions.Categorical(legal_probs)
            action = action_dist.sample()
            return action.item()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        # print(states.shape)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        terminated = torch.tensor(transition_dict['terminated'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        truncated = torch.tensor(transition_dict['truncated'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        
        mask = 1 - torch.logical_or(terminated, truncated).float()

        td_target = rewards + self.gamma * self.critic(next_states) * mask
        td_delta = td_target - self.critic(states)
        advantage = compute_advantage(self.gamma, self.lmbda,
                                               td_delta.cpu()).to(self.device)
        old_log_probs = torch.log(self.actor(states).gather(1,
                                                            actions)).detach()

        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps,
                                1 + self.eps) * advantage  # 截断
            actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数
            critic_loss = torch.mean(
                F.mse_loss(self.critic(states), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()