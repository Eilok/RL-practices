import numpy as np
def test_agent(agent, env, n_episode):
    ''' 测试agent在环境中的表现，返回平均回报 '''
    return_list = []
    for episode in range(n_episode):
        episode_return = 0
        state, info = env.reset()
        terminated = False
        truncated = False
        while not terminated and not truncated:
            action = agent.take_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            state = next_state
            episode_return += reward
        return_list.append(episode_return)
    return np.mean(return_list)