class OptionHedgingEnv(gym.Env):
    def __init__(self, data, xi=0.1, transaction_cost_proportion=0.5):
        super(OptionHedgingEnv, self).__init__()
        self.data = data.reset_index(drop=True)
        self.current_step = 0
        self.action_space = gym.spaces.Discrete(3) 
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(data.shape[1] - 3,), dtype=np.float32)
        self.position = 0 
        self.xi = xi  
        self.transaction_cost_proportion = transaction_cost_proportion  
        self.wealth = 0.0 

    def reset(self):
        self.current_step = 0
        self.position = 0
        self.wealth = 0.0
        return self.data.iloc[self.current_step].drop(['last', 'bid', 'ask']).values

    def step(self, action):
        prev_price = self.data.iloc[self.current_step]['last']
        prev_bid_price = self.data.iloc[self.current_step]['bid']
        prev_ask_price = self.data.iloc[self.current_step]['ask']

        self.current_step += 1
        if self.current_step >= len(self.data):
            self.current_step = 0

        obs = self.data.iloc[self.current_step].drop(['last', 'bid', 'ask']).values
        curr_price = self.data.iloc[self.current_step]['last']
        curr_bid_price = self.data.iloc[self.current_step]['bid']
        curr_ask_price = self.data.iloc[self.current_step]['ask']

        bid_ask_spread = curr_ask_price - curr_bid_price

        pnl = 0
        transaction_cost = 0

        if action == 0:  # Buy
            transaction_cost = self.transaction_cost_proportion * bid_ask_spread
            pnl = curr_price - curr_ask_price - transaction_cost
            self.position = 1

        elif action == 2:  # Sell
            transaction_cost = self.transaction_cost_proportion * bid_ask_spread
            pnl = curr_bid_price - curr_price - transaction_cost
            self.position = -1

        elif action == 1:  # Hold
            if self.position == 1:
                pnl = curr_price - prev_price
            elif self.position == -1:
                pnl = prev_price - curr_price

        self.wealth += pnl 

        reward = pnl - self.xi * abs(pnl)

        done = self.current_step == len(self.data) - 1
        return obs, reward, done, {}


# Define the DQN Model
def build_dqn_model(input_shape, output_shape):
    model = models.Sequential()
    model.add(layers.Dense(16, input_shape=input_shape, activation='relu'))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(output_shape, activation='linear'))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='mse')
    return model

# Set Replay Memory
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

train_env = OptionHedgingEnv(train_data, xi=0.1, transaction_cost_proportion=0.5)
validation_env = OptionHedgingEnv(validation_data, xi=0.1, transaction_cost_proportion=0.5)
test_env = OptionHedgingEnv(test_data, xi=0.1, transaction_cost_proportion=0.5)


input_shape = (train_env.observation_space.shape[0],)
output_shape = train_env.action_space.n
model = build_dqn_model(input_shape, output_shape)

# Replay Memory
memory = ReplayMemory(80)

gamma = 0.95
epsilon = 1.0
epsilon_min = 0.1
epsilon_decay = 0.995
batch_size = 64
num_episodes = 50

# Action selection
def select_action(state, epsilon):
    if np.random.rand() <= epsilon:
        return train_env.action_space.sample()
    q_values = model.predict(np.expand_dims(state, axis=0), verbose=0)
    return np.argmax(q_values[0])

# Optimize the model
def optimize_model():
    if len(memory) < batch_size:
        return
    transitions = memory.sample(batch_size)
    batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)

    batch_state = np.array(batch_state, dtype=np.float32)
    batch_action = np.array(batch_action, dtype=np.int32)
    batch_reward = np.array(batch_reward, dtype=np.float32)
    batch_next_state = np.array(batch_next_state, dtype=np.float32)
    batch_done = np.array(batch_done, dtype=np.float32)

    current_q_values = model.predict(batch_state, verbose=0)
    next_q_values = model.predict(batch_next_state, verbose=0)
    target_q_values = current_q_values.copy()

    for i in range(batch_size):
        target_q_values[i, batch_action[i]] = batch_reward[i] + (1 - batch_done[i]) * gamma * np.amax(next_q_values[i])

    model.train_on_batch(batch_state, target_q_values)

# Training phase
for episode in range(num_episodes):
    # Training phase
    state = train_env.reset()
    total_reward_train = 0
    while True:
        action = select_action(state, epsilon)
        next_state, reward, done, _ = train_env.step(action)
        memory.push((state, action, reward, next_state, done))
        state = next_state
        total_reward_train += reward
        optimize_model()
        if done:
            break
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # Validation phase
    state_val = validation_env.reset()
    total_reward_val = 0
    validation_rewards = []
    actual_val_rewards = []
    predicted_val_rewards = []
    
    while True:
        q_values_val = model.predict(np.expand_dims(state_val, axis=0), verbose=0)
        action_val = np.argmax(q_values_val[0])
        next_state_val, reward_val, done_val, _ = validation_env.step(action_val)
        total_reward_val += reward_val
        validation_rewards.append(total_reward_val)
        actual_val_rewards.append(reward_val)
        predicted_val_rewards.append(np.max(q_values_val[0]))
        state_val = next_state_val
        if done_val:
            break
    
    print(f"Episode {episode + 1}, Training Reward: {total_reward_train}, Validation Reward: {total_reward_val}")

# Testing phase
state_test = test_env.reset()
total_reward_test = 0
cumulative_test_rewards = []
actual_test_rewards = []
predicted_test_rewards = []

while True:
    q_values_test = model.predict(np.expand_dims(state_test, axis=0), verbose=0)
    action_test = np.argmax(q_values_test[0])
    next_state_test, reward_test, done_test, _ = test_env.step(action_test)
    total_reward_test += reward_test
    cumulative_test_rewards.append(total_reward_test)
    actual_test_rewards.append(reward_test)
    predicted_test_rewards.append(np.max(q_values_test[0]))
    state_test = next_state_test
    if done_test:
        break
