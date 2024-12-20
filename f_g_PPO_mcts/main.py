import torch
import gymnasium as gym

from PPO_F.env import FoxGooseEnv
from PPO_F.PPO import PPO
from PPO_F.train import train_on_policy_agent
from PPO_F.mul_train import mul_train_on_policy_agent
from utils import plot_return, load_agent

# 设置参数
actor_lr = 5e-5
critic_lr = 1e-5
num_episodes = 1000 # 训练的棋局数
# hidden_dim = 128
input_channels = 1 # 棋盘为单通道
gamma = 0.98
lmbda = 0.95
epochs = 20 # 训练轮数，比如下500盘棋，设置10轮，则每轮就是500/10=50盘棋子
train_epochs = 10 # 一盘棋子用来训练的轮数
eps = 0.2
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")
num_threads = 10

# 设置环境
env_name = 'Fox_and_Goose-v0'
gym.register(
    id=env_name,
    entry_point=FoxGooseEnv
)
env = gym.make(env_name)

# 开始训练
torch.manual_seed(0)
state_dim = env.observation_space.shape # (7, 7)
action_dim = env.action_space.n # 17

# 第一次训练
fox_agent = PPO(input_channels, action_dim, actor_lr, critic_lr, lmbda,
            train_epochs, eps, gamma, device)

# 在原基础上继续练
# fox_agent = PPO(input_channels, action_dim, actor_lr, critic_lr, lmbda,
#             train_epochs, eps, gamma, device)
# fox_agent = load_agent(fox_agent)

return_list = mul_train_on_policy_agent(env, fox_agent, num_threads, num_episodes, epochs, device, timeLimit=1000, iterationLimit=None)
# return_list = train_on_policy_agent(env, fox_agent, num_episodes, epochs, timeLimit=1000, iterationLimit=None)

plot_return(return_list, 'PPO', env_name)
