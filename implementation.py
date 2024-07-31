train_env = OptionHedgingEnv(train_data)
test_env = OptionHedgingEnv(reduced_test_features_df)

input_shape = (train_env.observation_space.shape[0],)
output_shape = train_env.action_space.n
model = build_dqn_model(input_shape, output_shape)

# Replay Memory
memory = ReplayMemory(20)

# Training Parameters
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
batch_size = 64
num_episodes = 10

# Action selection
def select_action(state, epsilon):
    if np.random.rand() <= epsilon:
        return train_env.action_space.sample()
    q_values = model.predict(np.expand_dims(state, axis=0))
    return np.argmax(q_values[0])

# Optimize the model
def optimize_model():
    if len(memory) < batch_size:
        return
    transitions = memory.sample(batch_size)
    batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)

    batch_state = np.array(batch_state)
    batch_action = np.array(batch_action)
    batch_reward = np.array(batch_reward)
    batch_next_state = np.array(batch_next_state)
    batch_done = np.array(batch_done, dtype=np.float32)

    current_q_values = model.predict(batch_state)
    next_q_values = model.predict(batch_next_state)
    target_q_values = current_q_values.copy()

    for i in range(batch_size):
        target_q_values[i, batch_action[i]] = batch_reward[i] + (1 - batch_done[i]) * gamma * np.amax(next_q_values[i])

    model.train_on_batch(batch_state, target_q_values)

# Train the model
for episode in range(num_episodes):
    state = train_env.reset()
    total_reward = 0
    for t in range(len(train_data)):
        action = select_action(state, epsilon)
        next_state, reward, done, _ = train_env.step(action)
        memory.push((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward
        optimize_model()
        if done:
            break
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    print(f"Episode {episode + 1}, Total Reward: {total_reward}")

# Evaluate the model
state = test_env.reset()
total_rewards = 0
actual_rewards = []
predicted_rewards = []
for _ in range(len(reduced_test_features_df)):
    q_values = model.predict(np.expand_dims(state, axis=0))
    action = np.argmax(q_values[0])
    next_state, reward, done, _ = test_env.step(action)
    total_rewards += reward
    actual_rewards.append(reward)
    predicted_rewards.append(np.max(q_values[0]))
    state = next_state
    if done:
        break

mse = mean_squared_error(actual_rewards, predicted_rewards)
print(f"Total Rewards during evaluation: {total_rewards}")
print(f"MSE during evaluation: {mse}")
