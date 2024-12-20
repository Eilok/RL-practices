from tqdm import tqdm
import numpy as np
import logging

from MCTS_G.Game import Fox_and_Goose, GooseAction, FoxAction, print_board_with_labels
from MCTS_G.MCTS import mcts
from utils import update_board, save_agent

# 配置日志记录
logging.basicConfig(
    filename='training_log.log',  # 日志文件名
    filemode='w',  # 写入模式
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO  # 日志级别
)

board = [
    [' ',' ','.','.','.',' ',' '],
    [' ',' ','.','.','.',' ',' '],
    ['.','.','.','.','.','.','.'],
    ['G','.','.','F','.','.','G'],
    ['G','G','G','G','G','G','G'],
    [' ',' ','G','G','G',' ',' '],
    [' ',' ','G','G','G',' ',' ']
]


def train_on_policy_agent(env, agent, num_episodes, epochs, timeLimit=None, iterationLimit=None):
    return_list = []
    for i in range(epochs):
        with tqdm(total=int(num_episodes/epochs), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/epochs)):
                # 初始化棋局
                episode_return = 0
                transition_dict = {
                    'states': [], 
                    'actions': [], 
                    'next_states': [], 
                    'rewards': [], 
                    'terminated': [], 
                    'truncated': []
                    }
                state, info = env.reset()
                terminated = False
                truncated = False

                # 针对鹅的环境
                s_G = Fox_and_Goose(board)
                tree = mcts(timeLimit, iterationLimit)

                logging.info(f'Round {i_episode}')
                
                while not terminated and not truncated:
                    # 鹅先走
                    G_action = tree.search(initialState=s_G)
                    s_G = s_G.takeAction(G_action)
                    G_moves = [[G_action.from_x, G_action.from_y], [G_action.to_x, G_action.to_y]] # 获取鹅的移动位置
                    logging.info(f'Goose action: {G_moves}')
                    
                    # 更新环境棋盘
                    env.unwrapped.board = update_board(env.unwrapped.board, G_moves)
                    state = env.unwrapped.get_obs()

                    # 增加通道维度
                    state = np.expand_dims(state, axis=0)
                    # print(state.shape)

                    # 记录狐狸移动
                    F_moves = [env.unwrapped.position]

                    # 采样狐狸的移动
                    # TODO: 让狐狸能够在一轮中移动多步
                    action = agent.take_action(state)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    next_state = np.expand_dims(next_state, axis=0)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['terminated'].append(terminated)
                    transition_dict['truncated'].append(truncated)
                    # state = next_state
                    episode_return += reward
                    
                    # 在鹅的环境中更新棋盘(狐狸移动)
                    F_moves.append(env.unwrapped.position)
                    # print(F_moves)
                    F_action = FoxAction(-1, F_moves)
                    s_G = s_G.takeAction(F_action)
                    logging.info(f'Fox action: {F_moves}')
                logging.info("The final board:")
                logging.info(s_G.board)
                logging.info(f"Round {i_episode} return: {episode_return}")

                return_list.append(episode_return)
                agent.update(transition_dict)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)

        # 每一个大轮存一次参数
        save_agent(agent)

    return return_list