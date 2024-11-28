####### 股票量化交易环境设计 ######

import gymnasium as gym
import numpy as np

# 环境设计
class StockTradingEnv_v0(gym.Env):
    def __init__(self, data, initial_balance=10000):
        super(StockTradingEnv_v0, self).__init__()

        self.data = data
        # self.seed = seed
        self.initial_balance = initial_balance
        self.current_step = 0

        # 状态空间
        self.observation_space = gym.spaces.Box(
            low=np.array([-1, 0], dtype=np.float32), 
            high=np.array([1, 1], dtype=np.float32), 
            dtype=np.float32
            )
        
        # 动作空间
        self.action_space = gym.spaces.Discrete(5)
        
        self.balance = initial_balance # 初始资金
        self.shares_held = 0 # 持有股票份额

    def _get_obs(self):
        ''' 获取当前时间步的状态 '''
        # 获取当前时间步的价格数据
        cur_price = self.data.loc[self.current_step, 'close']
        open_price = self.data.loc[self.current_step, 'open']
        high_price = self.data.loc[self.current_step, 'high']
        low_price = self.data.loc[self.current_step, 'low']

        # 计算价格变化率
        price_change = (cur_price - open_price) / open_price

        # 计算价格波动范围
        price_range = (high_price - low_price) / low_price

        return np.array([price_change, price_range], dtype=np.float32)
    

    def _take_action(self, action):
        ''' 执行动作 '''
        current_price = self.data.loc[self.current_step, 'close']

        # 卖出全部
        if action == 0:
            if self. shares_held > 0:
                self.balance += current_price * self.shares_held
                self.shares_held = 0
        # 卖出一半
        elif action == 1:
            if self.shares_held > 0:
                shares_sold = self.shares_held // 2
                self.balance += current_price * shares_sold
                self.shares_held -= shares_sold
        # 买入一半可用资金
        elif action == 3:
            available_amount = self.balance // 2
            shares_buy = available_amount // current_price
            if shares_buy > 0:
                self.shares_held += shares_buy
                self.balance -= shares_buy * current_price
        # 买入全部可用资金
        elif action == 4:
            available_amount = self.balance
            shares_buy = available_amount // current_price
            if shares_buy > 0:
                self.shares_held += shares_buy
                self.balance -= shares_buy * current_price

    def step(self, action):
        ''' 执行动作并返回下一个状态、奖励、是否终止、其他信息 '''
        # 获取当前价格
        current_price = self.data.loc[self.current_step, "close"]
        
        # 计算当前净值（在执行action之前）
        prev_net_worth = self.balance + self.shares_held * current_price
        
        # 执行交易
        self._take_action(action)
        
        # 移动到下一步
        self.current_step += 1
        
        # 获取下一步价格和净值
        next_price = self.data.loc[self.current_step, "close"]
        current_net_worth = self.balance + self.shares_held * next_price

        # 计算奖励
        reward = self._calculate_reward(prev_net_worth, current_net_worth)

        done = self.current_step == len(self.data) - 1
        truncated = False
        obs = self._get_obs()

        return obs, reward, done, truncated, {
            'net_worth': current_net_worth,
            'prev_net_worth': prev_net_worth,
            'reward': reward
        }


    def _calculate_reward(self, prev_net_worth, current_net_worth):
        # 基础收益率奖励
        profit_reward = (current_net_worth - prev_net_worth) / prev_net_worth
        
        # 持仓风险惩罚
        current_price = self.data.loc[self.current_step, "close"]
        position_ratio = (self.shares_held * current_price) / current_net_worth
        risk_penalty = -abs(position_ratio - 0.5) * 0.1  # 偏离50%仓位的惩罚
        
        # 破产惩罚
        if current_net_worth < self.initial_balance * 0.2:  # 资金低于初始资金20%
            bankrupt_penalty = -1
        else:
            bankrupt_penalty = 0
            
        # 合并奖励
        total_reward = profit_reward + risk_penalty + bankrupt_penalty
        
        # 限制奖励范围
        return np.clip(total_reward, -1, 1)
    
    def _get_info(self):
        return {
            'balance': self.balance,
            'shares_held': self.shares_held,
            'current_price': self.data.loc[self.current_step, "close"]
        }

    def reset(self, seed=None, options=None):
        ''' 重置环境 '''
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.prev_net_worth = self.initial_balance
        obs = self._get_obs()
        info = self._get_info()
        return obs, info
    

# 环境版本2：修改奖励函数
class StockTradingEnv_v1(gym.Env):
    def __init__(self, data, initial_balance=10000):
        super(StockTradingEnv_v1, self).__init__()

        self.data = data
        # self.seed = seed
        self.initial_balance = initial_balance
        self.current_step = 0

        # 状态空间
        self.observation_space = gym.spaces.Box(
            low=np.array([-1, 0], dtype=np.float32), 
            high=np.array([1, 1], dtype=np.float32), 
            dtype=np.float32
            )
        
        # 动作空间
        self.action_space = gym.spaces.Discrete(5)
        
        self.balance = initial_balance # 初始资金
        self.shares_held = 0 # 持有股票份额

    def _get_obs(self):
        ''' 获取当前时间步的状态 '''
        # 获取当前时间步的价格数据
        cur_price = self.data.loc[self.current_step, 'close']
        open_price = self.data.loc[self.current_step, 'open']
        high_price = self.data.loc[self.current_step, 'high']
        low_price = self.data.loc[self.current_step, 'low']

        # 计算价格变化率
        price_change = (cur_price - open_price) / open_price

        # 计算价格波动范围
        price_range = (high_price - low_price) / low_price

        return np.array([price_change, price_range], dtype=np.float32)
    

    def _take_action(self, action):
        ''' 执行动作 '''
        current_price = self.data.loc[self.current_step, 'close']

        # 卖出全部
        if action == 0:
            if self. shares_held > 0:
                self.balance += current_price * self.shares_held
                self.shares_held = 0
        # 卖出一半
        elif action == 1:
            if self.shares_held > 0:
                shares_sold = self.shares_held // 2
                self.balance += current_price * shares_sold
                self.shares_held -= shares_sold
        # 买入一半可用资金
        elif action == 3:
            available_amount = self.balance // 2
            shares_buy = available_amount // current_price
            if shares_buy > 0:
                self.shares_held += shares_buy
                self.balance -= shares_buy * current_price
        # 买入全部可用资金
        elif action == 4:
            available_amount = self.balance
            shares_buy = available_amount // current_price
            if shares_buy > 0:
                self.shares_held += shares_buy
                self.balance -= shares_buy * current_price

    def step(self, action):
        ''' 执行动作并返回下一个状态、奖励、是否终止、其他信息 '''
        # 获取当前价格
        current_price = self.data.loc[self.current_step, "close"]
        
        # 计算当前净值（在执行action之前）
        prev_net_worth = self.balance + self.shares_held * current_price
        
        # 执行交易
        self._take_action(action)
        
        # 移动到下一步
        self.current_step += 1
        
        # 获取下一步价格和净值
        next_price = self.data.loc[self.current_step, "close"]
        current_net_worth = self.balance + self.shares_held * next_price

        # 计算奖励
        reward = self._calculate_reward(prev_net_worth, current_net_worth)

        done = self.current_step == len(self.data) - 1
        truncated = False
        obs = self._get_obs()

        return obs, reward, done, truncated, {
            'net_worth': current_net_worth,
            'prev_net_worth': prev_net_worth,
            'reward': reward
        }


    def _calculate_reward(self, prev_net_worth, current_net_worth):
        reward = current_net_worth - prev_net_worth
        return reward
    
    def _get_info(self):
        return {
            'balance': self.balance,
            'shares_held': self.shares_held,
            'current_price': self.data.loc[self.current_step, "close"]
        }

    def reset(self, seed=None, options=None):
        ''' 重置环境 '''
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.prev_net_worth = self.initial_balance
        obs = self._get_obs()
        info = self._get_info()
        return obs, info