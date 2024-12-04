from tqdm import tqdm
import numpy as np

from professional.PPO import PPO
from GAIL.GAIL import GAIL

def run_GAIL(env, 
             state_dim, hidden_dim, action_dim, 
             actor_lr, critic_lr, lmbda, epochs, eps, gamma, device, lr_d,
             n_episode, expert_s, expert_a):
    
    agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda,
                    epochs, eps, gamma, device)
    gail = GAIL(agent, state_dim, action_dim, hidden_dim, lr_d, device)

    return_list = []
    with tqdm(total=n_episode, desc='Iteration') as pbar:
        for i in range(n_episode):
            episode_return = 0
            state, info = env.reset()
            terminated = False
            truncated = False
            state_list = []
            action_list = []
            next_state_list = []
            terminated_list = []
            # 采样一条episode
            while not terminated and not truncated:
                action = agent.take_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                state_list.append(state)
                action_list.append(action)
                next_state_list.append(next_state)
                terminated_list.append(terminated)
                episode_return += reward
                state = next_state
            return_list.append(episode_return)
            gail.learn(expert_s, expert_a, state_list, action_list, next_state_list, terminated_list)
            if (i+1) % 10 == 0:
                pbar.set_postfix({'return': '%.3f' % np.mean(return_list[-10:])})
            pbar.update(1)
    return return_list, gail