import torch
import matplotlib.pyplot as plt

def compute_advantage(gamma, lmbda, td_delta):
    ''' 优势函数 '''
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)

def save_agent(agent):
    ''' 保存模型 '''
    actor_path = './agents/actor.pth'
    critic_path = './agents/critic.pth'
    actor_optimizer_path = './agents/actor_optimizer.pth'
    critic_optimizer_path = './agents/critic_optimizer.pth'
    torch.save(agent.actor.state_dict(), actor_path)
    torch.save(agent.critic.state_dict(), critic_path)
    torch.save(agent.actor_optimizer.state_dict(), actor_optimizer_path)
    torch.save(agent.critic_optimizer.state_dict(), critic_optimizer_path)

def load_agent(agent):
    ''' 加载模型 '''
    actor_path = './agents/actor.pth'
    critic_path = './agents/critic.pth'
    actor_optimizer_path = './agents/actor_optimizer.pth'
    critic_optimizer_path = './agents/critic_optimizer.pth'
    agent.actor.load_state_dict(torch.load(actor_path))
    agent.critic.load_state_dict(torch.load(critic_path))
    agent.actor_optimizer.load_state_dict(torch.load(actor_optimizer_path))
    agent.critic_optimizer.load_state_dict(torch.load(critic_optimizer_path))
    return agent

def update_board(board, moves):
    ''' 
    根据鹅的动作更新棋盘 
    board -> ndarray
    '''
    from_pos = moves[0]
    to_pos = moves[1]
    board[from_pos[0], from_pos[1]] = '.'
    board[to_pos[0], to_pos[1]] = 'G'
    return board


def plot_return(returns, name, env_name):
    iteration_list = list(range(len(returns)))
    plt.plot(iteration_list, returns)
    plt.xlabel('Iterations')
    plt.ylabel('Returns')
    plt.title('{} on {}'.format(name, env_name))
    plt.show()