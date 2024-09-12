class OptionHedgingEnv(gym.Env):
    def __init__(self, data, xi=0.1, transaction_cost_proportion=0.5):
        super(OptionHedgingEnv, self).__init__()
        self.data = data.reset_index(drop=True)
        self.current_step = 0
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
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

        # Calculate the bid-ask spread
        bid_ask_spread = curr_ask_price - curr_bid_price

        # Initialise P&L and Transaction Cost
        pnl = 0
        transaction_cost = 0

        if action < -0.5:  # Buy
            transaction_cost = self.transaction_cost_proportion * bid_ask_spread
            pnl = curr_price - curr_ask_price - transaction_cost
            self.position = 1

        elif action > 0.5:  # Sell
            transaction_cost = self.transaction_cost_proportion * bid_ask_spread
            pnl = curr_bid_price - curr_price - transaction_cost 
            self.position = -1

        elif -0.5 <= action <= 0.5:  # Hold
            if self.position == 1:
                pnl = curr_price - prev_price
            elif self.position == -1:
                pnl = prev_price - curr_price 

        self.wealth += pnl

        reward = pnl - self.xi * abs(pnl)

        done = self.current_step == len(self.data) - 1
        return obs, reward, done, {}


# Define the Actor-Critic Models
def build_actor(input_shape, action_dim):
    model = models.Sequential()
    model.add(layers.Dense(64, input_shape=input_shape, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    model.add(layers.Dense(action_dim, activation='tanh'))
    return model


def build_critic(input_shape, action_dim):
    state_input = layers.Input(shape=input_shape)
    action_input = layers.Input(shape=(action_dim,))
    concat = layers.Concatenate()([state_input, action_input])
    x = layers.Dense(64, activation='relu')(concat)
    x = layers.Dense(64, activation='relu')(x)
    output = layers.Dense(1)(x)
    return models.Model(inputs=[state_input, action_input], outputs=output)

# Hyperparameters and Initialisation
gamma = 0.95
tau = 0.01
actor_lr = 1e-3
critic_lr = 1e-3
buffer_size = 80
batch_size = 64
num_episodes = 50

train_env = OptionHedgingEnv(train_data, xi=0.1, transaction_cost_proportion=0.5)
validation_env = OptionHedgingEnv(validation_data, xi=0.1, transaction_cost_proportion=0.5)
test_env = OptionHedgingEnv(test_data, xi=0.1, transaction_cost_proportion=0.5)

input_shape = (train_env.observation_space.shape[0],)
action_dim = train_env.action_space.shape[0]

actor = build_actor(input_shape, action_dim)
critic = build_critic(input_shape, action_dim)
target_actor = build_actor(input_shape, action_dim)
target_critic = build_critic(input_shape, action_dim)

actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)
critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)

memory = deque(maxlen=buffer_size)

target_actor.set_weights(actor.get_weights())
target_critic.set_weights(critic.get_weights())

# Function to update target networks
def update_target_networks(tau):
    new_weights = []
    for target_weights, model_weights in zip(target_actor.weights, actor.weights):
        new_weights.append(tau * model_weights + (1 - tau) * target_weights)
    target_actor.set_weights(new_weights)

    new_weights = []
    for target_weights, model_weights in zip(target_critic.weights, critic.weights):
        new_weights.append(tau * model_weights + (1 - tau) * target_weights)
    target_critic.set_weights(new_weights)

def select_action(state, noise_scale=0.05):
    state = np.expand_dims(state, axis=0)
    action = actor(state, training=False).numpy()[0]
    noise = np.random.normal(scale=noise_scale, size=action_dim)  # Exploration noise
    return np.clip(action + noise, train_env.action_space.low, train_env.action_space.high)

def train():
    if len(memory) < batch_size:
        return
    mini_batch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*mini_batch)

    states = tf.convert_to_tensor(np.array(states, dtype=np.float32))
    actions = tf.convert_to_tensor(np.array(actions, dtype=np.float32))
    rewards = tf.convert_to_tensor(np.array(rewards, dtype=np.float32).reshape(-1, 1))
    next_states = tf.convert_to_tensor(np.array(next_states, dtype=np.float32))
    dones = tf.convert_to_tensor(np.array(dones, dtype=np.float32).reshape(-1, 1))

    # Critic update
    with tf.GradientTape() as tape:
        next_actions = target_actor(next_states, training=False)
        target_q_values = target_critic([next_states, next_actions], training=False)
        target_values = rewards + (1 - dones) * gamma * target_q_values
        critic_q_values = critic([states, actions], training=True)
        critic_loss = tf.reduce_mean(tf.square(target_values - critic_q_values))
    critic_grads = tape.gradient(critic_loss, critic.trainable_variables)
    critic_optimizer.apply_gradients(zip(critic_grads, critic.trainable_variables))

    # Actor update
    with tf.GradientTape() as tape:
        actions = actor(states, training=True)
        actor_loss = -tf.reduce_mean(critic([states, actions], training=False))
    actor_grads = tape.gradient(actor_loss, actor.trainable_variables)
    actor_optimizer.apply_gradients(zip(actor_grads, actor.trainable_variables))

    update_target_networks(tau)

validation_cumulative_rewards = []  
testing_cumulative_rewards = []     

cumulative_reward_val = 0 
cumulative_reward_test = 0  

for episode in range(num_episodes):
    # Training phase
    state = train_env.reset()
    total_reward_train = 0
    while True:
        action = select_action(state, noise_scale=0.05)
        next_state, reward, done, _ = train_env.step(action)
        memory.append((state, action, reward, next_state, done))
        state = next_state
        total_reward_train += reward
        train()
        if done:
            break

    # Validation phase
    state_val = validation_env.reset()
    while True:
        action_val = select_action(state_val, noise_scale=0.01)
        next_state_val, reward_val, done_val, _ = validation_env.step(action_val)
        cumulative_reward_val += reward_val 
        validation_cumulative_rewards.append(cumulative_reward_val) 
        state_val = next_state_val
        if done_val:
            break

    print(f"Episode {episode + 1}, Training Reward: {total_reward_train}, Validation Cumulative Reward: {cumulative_reward_val}")

# Testing phase
for _ in range(num_episodes):
    state_test = test_env.reset()
    while True:
        action_test = select_action(state_test, noise_scale=0)
        next_state_test, reward_test, done_test, _ = test_env.step(action_test)
        cumulative_reward_test += reward_test 
        testing_cumulative_rewards.append(cumulative_reward_test) 
        state_test = next_state_test
        if done_test:
            break
