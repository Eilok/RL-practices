# 在每轮迭代中，GAIL中的策略和环境交互，采样新的状态动作对。
# 基于专家数据和策略新采样的数据，首先训练判别器，
# 然后将判别器的输出转换为策略的奖励信号，指导策略用PPO算法做训练。

from GAIL.Discriminator import Discriminator

import torch
import torch.nn.functional as F
import torch.nn as nn

class GAIL:
    ''' 生成对抗式模仿学习 '''
    def __init__(self, agent, state_dim, action_dim, hidden_dim, lr_d, device):
        self.device = device
        self.discriminator = Discriminator(state_dim, action_dim, hidden_dim).to(self.device)
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=lr_d)
        self.agent = agent

    def learn(self, expert_s, expert_a, agent_s, agent_a, next_s, terminated):
        expert_states = torch.tensor(expert_s, dtype=torch.float32).to(self.device)
        expert_actions = torch.tensor(expert_a, dtype=torch.int64).to(self.device)
        agent_states = torch.tensor(agent_s, dtype=torch.float32).to(self.device)
        agent_actions = torch.tensor(agent_a, dtype=torch.int64).to(self.device)
        # 将动作转换为独热编码
        expert_actions = F.one_hot(expert_actions, num_classes=2).float()
        agent_actions = F.one_hot(agent_actions, num_classes=2).float()

        expert_prob = self.discriminator(expert_states, expert_actions)
        agent_prob = self.discriminator(agent_states, agent_actions)

        # 使得agent接近1，专家接近0，从而计算损失
        discriminator_loss = nn.BCELoss()(agent_prob, torch.ones_like(agent_prob)) + nn.BCELoss()(expert_prob, torch.zeros_like(expert_prob))
        self.discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        self.discriminator_optimizer.step()
        
        # 将概率取负作为奖励，使得agent要尽可能降低自己被识别的概率
        rewards = -torch.log(agent_prob).detach().cpu().numpy()
        
        transition_dict = {
            'states': agent_s,
            'actions': agent_a,
            'rewards': rewards,
            'next_states': next_s,
            'dones': terminated
        }
        self.agent.update(transition_dict)
