import numpy as np
import torch
import gymnasium as gym
import random
import dill
import matplotlib.pyplot as plt

from behavior_clone.run import run_BC
from professional.run import run_PPO
from GAIL.run import run_GAIL


def sample_expert_data(n_episode, env, agent):
    states = []
    actions = []
    for episode in range(n_episode):
        state, info = env.reset()
        terminated = False
        truncated = False
        while not terminated and not truncated:
            action = agent.take_action(state)
            states.append(state)
            actions.append(action)
            next_state, reward, terminated, truncated, _ = env.step(action)
            state = next_state

        return np.array(states), np.array(actions)

def plot_return(returns, name, env_name):
    iteration_list = list(range(len(returns)))
    plt.plot(iteration_list, returns)
    plt.xlabel('Iterations')
    plt.ylabel('Returns')
    plt.title('{} on {}'.format(name, env_name))
    plt.show()

torch.manual_seed(0)
random.seed(0)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# 导入环境
env_name = 'CartPole-v1'
env = gym.make(env_name, render_mode='rgb_array')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
hidden_dim = 128

# 训练专家智能体(PPO)
PPO_params = {
    "env": env,
    "actor_lr": 1e-3,
    "critic_lr": 1e-2,
    "num_episodes": 200,
    "hidden_dim": hidden_dim,
    "gamma": 0.98,
    "lmbda": 0.95,
    "epochs": 10,
    "train_period": 50,
    "eps": 0.2,
    "device": device
}

PPO_returns, PPO_agent = run_PPO(**PPO_params)
plot_return(PPO_returns, 'PPO', env_name)
with open('./agents/PPO_agent.dill', 'wb') as f:
    dill.dump(PPO_agent, f)

# # 加载专家智能体(PPO)
# with open('./agents/PPO_agent.dill', 'rb') as f:
#     PPO_agent = dill.load(f)

# 采样专家数据
n_episode = 90
expert_s, expert_a = sample_expert_data(n_episode, env, PPO_agent)
n_samples = 30
random_index = random.sample(range(expert_s.shape[0]), n_samples)
expert_s = expert_s[random_index]
expert_a = expert_a[random_index]


# 使用BC训练智能体
BC_params = {
    'env': env,
    'state_dim': state_dim,
    'hidden_dim': hidden_dim,
    'action_dim': action_dim,
    'expert_s': expert_s,
    'expert_a': expert_a,
    'lr': 1e-3,
    'n_iterations': 1000,
    'batch_size': 64,
    'device': device
}

BC_returns, BC_agent = run_BC(**BC_params)
plot_return(BC_returns, 'BC', env_name)

with open('./agents/BC_agent.dill', 'wb') as f:
    dill.dump(BC_agent, f)

# 使用GAIL训练智能体
GAIL_params = {
    'env': env,
    'state_dim': state_dim,
    'hidden_dim': hidden_dim,
    'action_dim': action_dim,
    'actor_lr': 1e-3,
    'critic_lr': 1e-2,
    'lmbda': 0.95,
    'epochs': 10,
    'eps': 0.2,
    'gamma': 0.98,
    'device': device,
    'lr_d': 1e-3,
    'n_episode': 500, # 训练gail的轮数
    'expert_s': expert_s,
    'expert_a': expert_a,
}

GAIL_returns, GAIL_agent = run_GAIL(**GAIL_params)
plot_return(GAIL_returns, 'GAIL', env_name)
with open('./agents/GAIL_agent.dill', 'wb') as f:
    dill.dump(GAIL_agent, f)

