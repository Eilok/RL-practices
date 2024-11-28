import torch
import gymnasium as gym
import random
import numpy as np
import pandas as pd
import pickle

from environments import StockTradingEnv_v0, StockTradingEnv_v1
from ReplayBuffer import ReplayBuffer
from DQN import DQN
from train import train_agent
from plot import plot_return, plot_trades
from test import test

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

lr = 1e-3
num_episodes = 250
hidden_dim = 128
gamma = 0.98
epsilon = 0.01
target_updata = 20 # 目标网络更新频率
buffer_size = 10000 # 经验池大小
minimal_size = 100 # 经验池中数据量大于minimal_size时开始训练
batch_size = 64 # 经验池中采样数据量
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

env_name = "gymnasium_env/StockTrading-v1"

if __name__ == '__main__':
    data = pd.read_csv('./../data.csv')
    gym.register(
        id=env_name,
        entry_point=StockTradingEnv_v1,
    )
    env = gym.make(env_name, data=data)
    replay_buffer = ReplayBuffer(buffer_size)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_updata, device)
    
    agent, return_list = train_agent(env, agent, replay_buffer, num_episodes, minimal_size, batch_size)
    with open('./results/agent.pkl', 'wb') as f:
        pickle.dump(agent, f)
    plot_return(return_list, env)
    trade_history = test(env, agent)
    plot_trades(data, trade_history)
    