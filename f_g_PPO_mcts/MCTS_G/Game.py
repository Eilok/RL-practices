from copy import deepcopy

class GooseAction:
    ''' 鹅动作类 '''
    def __init__(self, player, from_x, from_y, to_x, to_y):
        self.player = player
        self.from_x = from_x
        self.from_y = from_y
        self.to_x = to_x
        self.to_y = to_y

    def __str__(self):
        return str(((self.from_x, self.from_y), (self.to_x, self.to_y)))
    
    def __repr__(self):
        return str(self)
    
    def __eq__(self, other):
        return (
            self.__class__ == other.__class__ 
                and self.from_x == other.from_x 
                and self.from_y == other.from_y
                and self.to_x == other.to_x 
                and self.to_y == other.to_y
                and self.player == other.player
                )

    def __hash__(self):
        return hash((self.from_x, self.from_y, self.to_x, self.to_y, self.player))


class FoxAction:
    ''' 狐狸动作类 '''
    def __init__(self, player, moves):
        self.player = player
        self.moves = moves

    def __str__(self):
        return str(self.moves)
    
    def __repr__(self):
        return str(self)


class Fox_and_Goose:
    ''' 游戏状态类 '''
    def __init__(self, board):
        self.board = board
        self.currentPlayer = 1 # 1: goose, -1: fox
        self.winner = None

    def getPossibleActions(self):
        ''' 
        获取所有可行动作
        返回：可行动作列表
        '''
        possibleActions = []

        # 移动方向，从上开始，顺时针选择
        actions = [[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]]
        
        # 当前玩家为goose
        if self.currentPlayer == 1:
            for i in range(len(self.board)):
                for j in range(len(self.board[i])):
                    if self.board[i][j] == 'G':
                        for action in actions:
                            new_x = i + action[0]
                            new_y = j + action[1]
                            moves = [[i, j], [new_x, new_y]]
                            
                            # 判断动作的合法性
                            if not self.is_valid_move(moves):
                                continue

                            possibleActions.append(GooseAction(self.currentPlayer, i, j, new_x, new_y))
        
        # 当前玩家为fox
        if self.currentPlayer == -1:
            for i in range(len(self.board)):
                for j in range(len(self.board[i])):
                    if self.board[i][j] == 'F':
                        # 考虑四周移动一步
                        for action in actions:
                            moves = [[i, j]]
                            new_x = i + action[0]
                            new_y = j + action[1]
                            if new_x >= 0 and new_x < len(self.board) and new_y >= 0 and new_y < len(self.board):
                            # 四周有鹅
                                if self.board[new_x][new_y] == 'G':
                                    flag = True # 是否可以抓鹅
                                    cap_x, cap_y = deepcopy(new_x), deepcopy(new_y)
                                    while flag:
                                        # 捉鹅后跳到的位置
                                        new_cap_x = cap_x+cap_x-i
                                        new_cap_y = cap_y+cap_y-j
                                        capture_pos = [new_cap_x, new_cap_y]
                                        # 验证捉鹅的合法性
                                        if capture_pos[0] >= 0 and capture_pos[0] < len(self.board) and capture_pos[1] >= 0 and capture_pos[1] < len(self.board):
                                            if self.board[capture_pos[0]][capture_pos[1]] == '.':
                                                moves.append(capture_pos)
                                                cap_x = new_cap_x
                                                cap_y = new_cap_y
                                                continue
                                        # 到这说明抓鹅失败，退出循环
                                        flag = False

                                    # 捕捉成功，添加动作
                                    if len(moves) > 1:
                                        possibleActions.append(FoxAction(self.currentPlayer, moves))
                                else:
                                    moves.append([new_x, new_y])
                                    if not self.is_valid_move(moves):
                                        continue
                                    possibleActions.append(FoxAction(self.currentPlayer, moves))
        
        return possibleActions

    def is_valid_move(self, moves):
        ''' 
        判断是否为合法走子
        参数：movs = [[x1, y1], [x2, y2]]
        返回：Ture/False
        '''
        from_pos = moves[0]
        to_pos = moves[1]
        # 超出棋盘
        if to_pos[0] < 0 or to_pos[0] >= len(self.board) or to_pos[1] < 0 or to_pos[1] >= len(self.board):
            return False
        # 走子有棋子占位/无效位
        if self.board[to_pos[0]][to_pos[1]] != '.':
            return False
        # 同余才可对角
        if (from_pos[0] != to_pos[0] and from_pos[1] != to_pos[1]) and ((from_pos[0] % 2) != (from_pos[1] % 2)):
            return False
        return True
    
    def takeAction(self, action):
        ''' 
        执行动作,并且更新棋盘状态
        参数：action类
        返回：执行动作后的新状态
        '''
        newState = deepcopy(self)
        # 鹅的动作
        if isinstance(action, GooseAction):
            newState.board[action.to_x][action.to_y] = 'G'
            newState.board[action.from_x][action.from_y] = '.'
        # 狐的动作
        if isinstance(action, FoxAction):
            fox_moves = action.moves
            for i in range(len(fox_moves)-1):
                # 抓鹅的动作处理
                if (abs(fox_moves[i+1][0] - fox_moves[i][0]) == 2) or (abs(fox_moves[i+1][1] - fox_moves[i][1]) == 2):
                    mid_x = (fox_moves[i][0] + fox_moves[i+1][0]) // 2
                    mid_y = (fox_moves[i][1] + fox_moves[i+1][1]) // 2
                    newState.board[fox_moves[i][0]][fox_moves[i][1]] = '.'
                    newState.board[mid_x][mid_y] = '.'
                    newState.board[fox_moves[i+1][0]][fox_moves[i+1][1]] = 'F'
                else:
                    newState.board[fox_moves[i][0]][fox_moves[i][1]] = '.'
                    newState.board[fox_moves[i+1][0]][fox_moves[i+1][1]] = 'F'
        newState.currentPlayer = self.currentPlayer * -1   
        return newState

    def isTerminal(self):
        '''
        判断游戏是否结束
        返回：Ture/False
        '''
        # Checks if the game is won by the fox
        goose_count = 0
        for i in range(len(self.board)):
            for j in range(len(self.board[i])):
                if self.board[i][j] == 'G':
                    goose_count += 1
        
        # If less than 4 geese left, the fox wins
        if goose_count < 4:
            self.winner = -1
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
        self.winner = 1
        return True

    def getReward(self):
        '''
        计算奖励值
        返回：赢则1，输或平局则0
        '''
        if self.winner != None:
            return 1
        return 0

def print_board_with_labels(board):
    # 打印列号
    column_labels = '   ' + ' '.join(str(i) for i in range(len(board[0])))
    print(column_labels)
    
    # 打印每行，并在前面加上行号
    for idx, row in enumerate(board):
        print(f"{idx}  " + ' '.join(row))

    