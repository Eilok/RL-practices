from MCTS_G.Game import Fox_and_Goose, FoxAction, print_board_with_labels
from MCTS_G.MCTS import mcts
from PPO_F.env import FoxGooseEnv
from utils import update_board, load_agent
import PPO_F.PPO
from PPO_F.PPO import PPO

import gymnasium as gym
import numpy as np
import torch


board = [
    [' ',' ','.','.','.',' ',' '],
    [' ',' ','.','.','.',' ',' '],
    ['.','.','.','.','.','.','.'],
    ['G','.','.','F','.','.','G'],
    ['G','G','G','G','G','G','G'],
    [' ',' ','G','G','G',' ',' '],
    [' ',' ','G','G','G',' ',' ']
]

def self_play(F_agent, device, timeLimit=None, iterationLimit=None):
    ''' PPO plays fox, MCTS plays goose '''
    # 初始化鹅环境
    G_env = Fox_and_Goose(board)
    tree = mcts(timeLimit=timeLimit, iterationLimit=iterationLimit)

    # 初始化狐环境
    env_name = 'Fox_and_Goose-v0'
    gym.register(
        id=env_name,
        entry_point=FoxGooseEnv
    )
    F_env = gym.make(env_name)
    F_state, _ = F_env.reset()
    terminated = False
    truncated = False

    total_return = 0

    print("initial board:")
    print_board_with_labels(G_env.board)

    step = 0

    while not terminated and not truncated:
        print()
        print(f"The round {step}:")

        # 鹅走子
        G_action = tree.search(initialState=G_env)
        G_env = G_env.takeAction(G_action)
        G_moves = [[G_action.from_x, G_action.from_y], [G_action.to_x, G_action.to_y]]
        print(f"goose move: {G_moves}")

        # 更新狐狸环境
        F_moves = [F_env.unwrapped.position]
        F_env.unwrapped.board = update_board(F_env.unwrapped.board, G_moves)
        F_state = F_env.unwrapped.get_obs()
        F_state = np.expand_dims(F_state, axis=0)
        
        # 狐狸走子
        mask = F_env.unwrapped.get_legal_actions(device)

        # print("掩码:", mask)
        F_action = F_agent.take_action(F_state, mask)
        F_next_state, reward, terminated, truncated, _ = F_env.step(F_action)
        total_return += reward
        F_moves.append(F_env.unwrapped.position)
        print(f"fox move: {F_moves}")

        # 更新鹅环境
        F_action = FoxAction(-1, F_moves)
        G_env = G_env.takeAction(F_action)

        print_board_with_labels(G_env.board)

        step += 1
    print(f"winner: {F_env.unwrapped.winner}")
    print(f"fox chess board: {F_env.unwrapped.board}")

if __name__ == '__main__':
    actor_lr = 1e-3
    critic_lr = 1e-2
    # hidden_dim = 128
    input_channels = 1 # 棋盘为单通道
    gamma = 0.98
    lmbda = 0.95
    epochs = 10
    eps = 0.2
    action_dim = 17
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")
    F_agent = PPO(input_channels, action_dim, actor_lr, critic_lr, lmbda,
            epochs, eps, gamma, device)
    F_agent = load_agent(F_agent)
    self_play(F_agent, device, timeLimit=4000)