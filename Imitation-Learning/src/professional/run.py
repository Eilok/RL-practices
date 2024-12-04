import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
import torch

from professional.PPO import PPO
import professional.rl_utils as rl_utils

def run_PPO(env, actor_lr, critic_lr, num_episodes, hidden_dim, gamma, lmbda, epochs, train_period, eps, device):

    env = RecordVideo(env, video_folder='./professional/video', name_prefix='training', 
                    episode_trigger=lambda x: x % train_period == 0)
    env = RecordEpisodeStatistics(env)
    torch.manual_seed(0)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    ppo_agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda,
                    epochs, eps, gamma, device)
    return_list, ppo_agent = rl_utils.train_on_policy_agent(env, ppo_agent, num_episodes)

    return return_list, ppo_agent