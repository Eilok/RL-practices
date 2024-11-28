###### 经验回放池类 ######

import collections
import random
import numpy as np

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity) # 双端队列->队列

    def add(self, state, action, reward, next_state, done):
        ''' 将数据加入经验池 '''
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        ''' 从经验池中随机采样batch_size个样本 '''
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done
    
    def size(self):
        ''' 返回经验池中数据大小 '''
        return len(self.buffer)