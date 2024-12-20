import numpy as np
import gymnasium as gym
import torch

class FoxGooseEnv(gym.Env):
    def __init__(self):
        super(FoxGooseEnv, self).__init__()

        self.grid_size = 7

        # 状态空间
        # 0：非法区，1：空格，2：鹅，3：狐狸
        self.observation_space = gym.spaces.Box(
            low=0, high=3, shape=(self.grid_size, self.grid_size), dtype=np.int32
        )

        self.board = np.array([
            [' ', ' ', '.', '.', '.', ' ', ' '],
            [' ', ' ', '.', '.', '.', ' ', ' '],
            ['.', '.', '.', '.', '.', '.', '.'],
            ['G', '.', '.', 'F', '.', '.', 'G'],
            ['G', 'G', 'G', 'G', 'G', 'G', 'G'],
            [' ', ' ', 'G', 'G', 'G', ' ', ' '],
            [' ', ' ', 'G', 'G', 'G', ' ', ' ']
        ])

        self._chess_to_num = {
            ' ': 0,
            '.': 1,
            'G': 2,
            'F': 3
        }

        # 动作空间
        self.action_space = gym.spaces.Discrete(17)

        # 动作映射
        self._action_to_direction = {
            0: [-1, 0],  # 上
            1: [-1, 1],  # 上右
            2: [0, 1],  # 右
            3: [1, 1],  # 下右
            4: [1, 0],  # 下
            5: [1, -1],  # 下左
            6: [0, -1],  # 左
            7: [-1, -1],  # 上左
            8: [-2, 0],  # 上上
            9: [-2, 2], # 上右两格
            10: [0, 2],  # 右两格
            11: [2, 2],  # 下右两格
            12: [2, 0],  # 下下
            13: [2, -2],  # 下左两格
            14: [0, -2],  # 左两格
            15: [-2, -2],  # 上左两格
            16: [0, 0]  # 静止
        }

        # 棋盘奖励
        self.chess_reward = np.array([
            [0, 0, -2, -4, -2, 0, 0],
            [0, 0, -2, -1, -2, 0, 0],
            [-2, -2, -1, -1, -1, -2, -2],
            [-4, -1, -1, -1, -1, -1, -4],
            [-2, -2, -1, -1, -1, -2, -2],
            [0, 0, -2, -1, -2, 0, 0],
            [0, 0, -2, -4, -2, 0, 0]
        ])

        # 狐狸的位置
        self.position = [3, 3]  

        self.winner = None
        self.steps = 0 # 记录步数

    def get_obs(self):
        ''' 获取当前状态 '''
        numeric_board = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                numeric_board[i, j] = self._chess_to_num[self.board[i][j]]

        return numeric_board
    
    def _get_info(self):
        ''' 获取额外信息 '''
        return {
            'position': self.position,
            'steps': self.steps
        }
    
    def step(self, action):
        ''' 执行动作(动作索引），返回下一个状态，奖励，截断，终止，额外信息'''
        # 获取移动
        direction = self._action_to_direction[action]
        next_position = [self.position[0] + direction[0], self.position[1] + direction[1]]
        moves = [self.position, next_position]
        
        self.steps += 1

        # 执行动作，更改棋盘

        # 抓鹅的动作
        if (abs(self.position[0] - next_position[0]) == 2) or (abs(self.position[1] - next_position[1]) == 2):
            mid_x = (self.position[0] + next_position[0]) // 2
            mid_y = (self.position[1] + next_position[1]) // 2
            self.board[self.position[0], self.position[1]] = '.'
            self.board[mid_x, mid_y] = '.'
            self.board[next_position[0], next_position[1]] = 'F'
        # 走一格
        else:
            self.board[self.position[0], self.position[1]] = '.'
            self.board[next_position[0], next_position[1]] = 'F'
        self.position = next_position
        
        # 判断是否终止
        truncated = self.is_truncated()
        terminated = self.is_terminated()
        # TODO：优化
        self.board = np.array(self.board)

        # 获取奖励
        reward = self._get_reward(moves)

        observation = self.get_obs()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info

    def _is_valid_move(self, moves):
        ''' 判断狐狸动作是否合法 '''
        from_pos = moves[0]
        to_pos = moves[1]

        # 检查目标位置是否在网格内
        if to_pos[0] < 0 or to_pos[0] >= self.grid_size or to_pos[1] < 0 or to_pos[1] >= self.grid_size:
            return False
        # 同余才可对角
        if (from_pos[0] != to_pos[0] and from_pos[1] != to_pos[1]) and ((from_pos[0] % 2) != (from_pos[1] % 2)):
            return False
        # 目标位置不为空
        if self.board[to_pos[0], to_pos[1]] != '.':
            return False
        # 跳两格时是否是抓鹅
        if (abs(from_pos[0] - to_pos[0]) == 2) or (abs(from_pos[1] - to_pos[1]) == 2):
            if self.board[from_pos[0] + (to_pos[0] - from_pos[0]) // 2, from_pos[1] + (to_pos[1] - from_pos[1]) // 2] != 'G':
               return False
        return True
    

    def get_legal_actions(self, device):
        ''' 生成合法动作的掩码 '''
        mask = np.zeros(self.action_space.n, dtype=np.float32)
        for action in range(self.action_space.n):
            to_pos = [self.position[0] + self._action_to_direction[action][0], self.position[1] + self._action_to_direction[action][1]]
            moves = [self.position, to_pos]
            if self._is_valid_move(moves):
                mask[action] = 1.0
        return torch.tensor(mask, device=device)

    def is_truncated(self):
        ''' 判断游戏是否提前截断 '''
        if self.steps >= 150:
            return True
        else:
            return False

    def is_terminated(self):
        '''
        判断游戏是否结束
        返回：Ture/False
        '''
        # TODO: 看怎么优化这个棋盘的获取
        self.board = self.board.tolist()

        # Checks if the game is won by the fox
        goose_count = 0
        for i in range(len(self.board)):
            for j in range(len(self.board[i])):
                if self.board[i][j] == 'G':
                    goose_count += 1
        
        # If less than 4 geese left, the fox wins
        if goose_count < 4:
            self.winner = 'F'
            return True

        # Checks if the game is won by the geese
        # 找狐狸的位置
        fox_pos = [0,0]
        for i in range(len(self.board)):
            for j in range(len(self.board[i])):
                if self.board[i][j] == 'F':
                    fox_pos = [i,j]
                    break
        
        # 判断狐狸是否还可以移动
        # Can the fox move?
        for i in range(max(fox_pos[0]-1,0),min(fox_pos[0]+2,len(self.board))):
            for j in range(max(fox_pos[1]-1,0),min(fox_pos[1]+2,len(self.board))):
            # 同行/同列/可对角移动
                if (i == fox_pos[0] or j == fox_pos[1]) or ((fox_pos[0] % 2) == (fox_pos[1] % 2)):
                    if self.board[i][j] == '.':
                        return False  # Fox can move one step
                    if self.board[i][j] == 'G':
                        capture_pos = [i+i-fox_pos[0],j+j-fox_pos[1]]
                        if capture_pos[0] >= 0 and capture_pos[0] < len(self.board) and capture_pos[1] >= 0 and capture_pos[1] < len(self.board):
                            if self.board[capture_pos[0]][capture_pos[1]] == '.':
                                return False  # Fox can capture a goose

        # The fox cannot move, the geese win
        self.winner = 'G'
        return True

    def _get_reward(self, moves):
        ''' 获取奖励 '''
        from_pos = moves[0]
        to_pos = moves[1]
        if self.winner == 'F':
            return 50
        elif self.winner == 'G':
            return -100
        elif (abs(from_pos[0] - to_pos[0]) == 2) or (abs(from_pos[1] - to_pos[1]) == 2):
            return 10 - self.chess_reward[to_pos[0], to_pos[1]]
        else:
            return self.chess_reward[to_pos[0], to_pos[1]]


    def reset(self, seed=None, options=None):
        ''' 重置环境 '''
        super().reset(seed=seed, options=options)

        self.board = np.array([
            [' ', ' ', '.', '.', '.', ' ', ' '],
            [' ', ' ', '.', '.', '.', ' ', ' '],
            ['.', '.', '.', '.', '.', '.', '.'],
            ['G', '.', '.', 'F', '.', '.', 'G'],
            ['G', 'G', 'G', 'G', 'G', 'G', 'G'],
            [' ', ' ', 'G', 'G', 'G', ' ', ' '],
            [' ', ' ', 'G', 'G', 'G', ' ', ' ']
        ])
        self.position = [3, 3]
        self.winner = None
        self.steps = 0
        obs = self.get_obs()
        info = self._get_info()
        return obs, info


# def update_board(board, moves):
#     ''' 
#     根据鹅的动作更新棋盘 
#     board -> ndarray
#     '''
#     from_pos = moves[0]
#     to_pos = moves[1]
#     board[from_pos[0], from_pos[1]] = ' '
#     board[to_pos[0], to_pos[1]] = 'G'
#     return board

# 测试
# if __name__ == '__main__':
#     env_name = 'Fox_and_Goose-v0'
#     gym.register(
#         id=env_name,
#         entry_point=FoxGooseEnv
#     )
#     env = gym.make(env_name)

    # print(env.reset())
    # action_dim = env.action_space.n
    # print(action_dim)
    # print(env.unwrapped.get_obs())
    # env.unwrapped.board = update_board(env.unwrapped.board, [[4,1],[3,1]])
    # print(env.unwrapped.get_obs())