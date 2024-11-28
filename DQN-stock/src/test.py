def test(env, agent, episodes=1):
    trade_history = []
    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        episode_return = 0
        

        while not done:
            # 获取当前信息
            current_price = env.unwrapped.data.loc[env.unwrapped.current_step, 'close']
            current_balance = env.unwrapped.balance
            current_shares = env.unwrapped.shares_held

            # 交易决策
            action = agent.take_action(state)

            # 记录交易信息
            trade_info = {
                'step': env.unwrapped.current_step,
                'price': current_price,
                'balance': current_balance,
                'shares': current_shares,
                'action': action,
                'net_worth': current_balance + current_shares * current_price,
            }
            trade_history.append(trade_info)

            # 执行交易
            next_state, reward, done, truncated, _ = env.step(action)
            state = next_state
            episode_return += reward

    return trade_history

