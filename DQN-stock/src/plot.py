import matplotlib.pyplot as plt
def plot_return(return_list, env_name):
    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DQN on {}'.format(env_name))
    plt.show()

def plot_trades(df, trade_history):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
    
    # 绘制价格走势
    ax1.plot(df['close'], label='Price')
    ax1.set_title('Price Movement and Trading Actions')
    
    # 标记买卖点
    for trade in trade_history:
        if trade['action'] == 0:  # 卖出全部
            ax1.scatter(trade['step'], trade['price'], 
                       color='green', marker='^', s=100)
        elif trade['action'] == 1:  # 卖出一半
            ax1.scatter(trade['step'], trade['price'],
                       color='lightgreen', marker='v', s=100)
        elif trade['action'] == 2:  # 持有
            ax1.scatter(trade['step'], trade['price'], 
                       color='blue', marker='o', s=100)
        elif trade['action'] == 3:  # 买入一半
            ax1.scatter(trade['step'], trade['price'],
                       color='#FFA07A', marker='<', s=100)
        elif trade['action'] == 4:  # 买入全部
            ax1.scatter(trade['step'], trade['price'],
                       color='red', marker='>', s=100)
    
    # 绘制持仓量变化
    steps = [t['step'] for t in trade_history]
    shares = [t['shares'] for t in trade_history]
    ax2.plot(steps, shares)
    ax2.set_title('Position Size')
    
    # 绘制账户价值变化
    net_worth = [t['net_worth'] for t in trade_history]
    ax3.plot(steps, net_worth)
    ax3.set_title('Account Value')
    
    plt.tight_layout()
    plt.show()