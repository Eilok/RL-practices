import torch
import numpy as np
from tqdm import tqdm

from behavior_clone.BehaviorClone import BehaviorClone
from behavior_clone.test_agent import test_agent

def run_BC(env, state_dim, hidden_dim, action_dim, 
           expert_s, expert_a, lr, n_iterations, batch_size, device):
    torch.manual_seed(0)
    np.random.seed(0)

    bc_agent = BehaviorClone(state_dim, hidden_dim, action_dim, lr, device)

    test_returns = []

    with tqdm(total=n_iterations, desc="进度条") as pbar:
        for i in range(n_iterations):
            sample_indices = np.random.randint(low=0,
                                            high=expert_s.shape[0],
                                            size=batch_size)
            bc_agent.learn(expert_s[sample_indices], expert_a[sample_indices])
            current_return = test_agent(bc_agent, env, 5)
            test_returns.append(current_return)
            if (i + 1) % 10 == 0:
                pbar.set_postfix({'return': '%.3f' % np.mean(test_returns[-10:])})
            pbar.update(1)
    return test_returns, bc_agent