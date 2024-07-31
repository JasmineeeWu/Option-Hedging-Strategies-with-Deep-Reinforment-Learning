# Define the Reinforcement Learning Environment
class OptionHedgingEnv(gym.Env):
    def __init__(self, data):
        super(OptionHedgingEnv, self).__init__()
        self.data = data.reset_index(drop=True)
        self.current_step = 0
        self.action_space = gym.spaces.Discrete(3)  # Buy, Hold, Sell
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(data.shape[1] - 1,), dtype=np.float32)
        self.position = 0  # Initial position: 0 (no position)
        self.cost = 0.01  # Transaction cost as proportion of bid-ask spread

    def reset(self):
        self.current_step = 0
        self.position = 0
        return self.data.iloc[self.current_step].drop('last').values

    def step(self, action):
        prev_price = self.data.iloc[self.current_step]['last']
        self.current_step += 1
        if self.current_step >= len(self.data):
            self.current_step = 0

        obs = self.data.iloc[self.current_step].drop('last').values
        curr_price = self.data.iloc[self.current_step]['last']
        bid_price = curr_price * (1 - self.cost / 2)
        ask_price = curr_price * (1 + self.cost / 2)

        # Calculate P&L
        pnl = 0
        transaction_cost = 0
        if action == 0:  # Buy
            transaction_cost = self.cost * (ask_price - bid_price)
            pnl = curr_price - ask_price - transaction_cost
            self.position = 1
        elif action == 2:  # Sell
            transaction_cost = self.cost * (ask_price - bid_price)
            pnl = bid_price - curr_price - transaction_cost
            self.position = -1
        elif action == 1:  # Hold
            pnl = (curr_price - prev_price) * self.position

        reward = pnl - transaction_cost
        done = self.current_step == len(self.data) - 1
        return obs, reward, done, {}
