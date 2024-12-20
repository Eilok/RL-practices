import random
import math
import time

def randomPolicy(state):
    ''' rollout时使用的方法：随机策略 '''
    while not state.isTerminal():
        try:
            action = random.choice(state.getPossibleActions())
        except IndexError:
            raise Exception("Non-terminal state has no possible actions: " + str(state))
        state = state.takeAction(action)
    return state.getReward()

class treeNode:
    ''' mcts节点类 '''
    def __init__(self, state, parent):
        self.state = state
        self.isTerminal = state.isTerminal()
        self.isFullyExpanded = self.isTerminal
        self.parent = parent
        self.numVisits = 0 # 访问次数
        self.totalReward = 0 # 胜场数
        self.children = {} # 孩子字典{action: node}

class mcts():
    def __init__(self, timeLimit=None, iterationLimit=None, explorationConstant=1 / math.sqrt(2),
                 rolloutPolicy=randomPolicy):
        if timeLimit != None:
            if iterationLimit != None:
                raise ValueError("Cannot have both a time limit and an iteration limit")
            # time taken for each MCTS search in milliseconds
            self.timeLimit = timeLimit
            self.limitType = 'time'
        else:
            if iterationLimit == None:
                raise ValueError("Must have either a time limit or an iteration limit")
            # number of iterations of the search
            if iterationLimit < 1:
                raise ValueError("Iteration limit must be greater than one")
            self.searchLimit = iterationLimit # 搜索深度限制
            self.limitType = 'iterations'
        self.explorationConstant = explorationConstant
        self.rollout = rolloutPolicy

    def search(self, initialState):
        ''' 向下搜索所有可能的状态 '''
        self.root = treeNode(initialState, None)

        # 时间限制
        if self.limitType == 'time':
            timeLimit = time.time() + self.timeLimit / 1000
            while time.time() < timeLimit:
                self.executeRound()
        # 次数限制
        else:
            for i in range(self.searchLimit):
                self.executeRound()
        # 前面执行完后，叶子节点就存放他们的信息
        bestChild = self.getBestChild(self.root, 0)
        return self.getAction(self.root, bestChild)

    def executeRound(self):
        ''' 执行一次模拟流程 '''
        node = self.selectNode(self.root)
        reward = self.rollout(node.state)
        self.backpropogate(node, reward)

    def selectNode(self, node):
        ''' 
        选择一个节点。
        1. 有子节点，获取UCT最大的节点
        2. 无子节点，使用expand扩展子节点
        '''
        # 一直找到游戏结束
        while not node.isTerminal:
            # 该节点已经完全拓展，选择最好的孩子
            if node.isFullyExpanded:
                node = self.getBestChild(node, self.explorationConstant)
            # 该节点没有完全拓展，扩展子节点
            else:
                return self.expand(node)
        return node
    
    def expand(self, node):
        ''' 扩展一个节点，返回一个扩展后的节点 '''
        actions = node.state.getPossibleActions()
        for action in actions:
            if action not in node.children:
                newNode = treeNode(node.state.takeAction(action), node)
                node.children[action] = newNode
                if len(actions) == len(node.children):
                    node.isFullyExpanded = True
                return newNode
        # expand 只有在需要拓展时才被调用    
        raise Exception("Should never reach here")
    
    def backpropogate(self, node, reward):
        ''' 反向更新统计信息 '''
        while node is not None:
            node.numVisits += 1
            node.totalReward += reward
            node = node.parent
    
    def getBestChild(self, node, explorationValue):
        '''  获取最优的child节点 '''
        bestValue = float("-inf")
        bestNodes = []
        for child in node.children.values():
            # 计算UCT值
            nodeValue = child.totalReward / child.numVisits + explorationValue * math.sqrt(
                2 * math.log(node.numVisits) / child.numVisits)
            if nodeValue > bestValue:
                bestValue = nodeValue
                bestNodes = [child]
            elif nodeValue == bestValue:
                bestNodes.append(child)
        return random.choice(bestNodes)

    def getAction(self, root, bestChild):
        ''' 从节点中动作和节点 '''
        for action, node in root.children.items():
            if node is bestChild:
                return action
