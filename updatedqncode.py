# Importing necessary libraries
import gym
import numpy as np
import pandas as pd
from collections import deque
import random
import tensorflow as tf
from tensorflow.keras import models, layers
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load and prepare your data here
# data = ...

# Split the data into training, validation, and test sets
train_data = data[data['date'] <= '2024-06-14']
validation_data = data[(data['date'] > '2024-06-14') & (data['date'] <= '2024-06-21')]
test_data = data[data['date'] > '2024-06-21']

# Preprocess the training data
train_data = train_data.drop(columns=['contractID', 'symbol', 'expiration', 'strike', 'date'])
features_train = train_data.drop(columns=['last', 'bid', 'ask'])
target_train = train_data['last']

scaler = StandardScaler()
scaled_features_train = scaler.fit_transform(features_train)

train_data_processed = pd.DataFrame(scaled_features_train, columns=features_train.columns)
train_data_processed['last'] = target_train.values
train_data_processed['bid'] = train_data['bid'].values
train_data_processed['ask'] = train_data['ask'].values

# Preprocess the validation data
validation_data = validation_data.drop(columns=['contractID', 'symbol', 'expiration', 'strike', 'date'])
features_validation = validation_data.drop(columns=['last', 'bid', 'ask'])
target_validation = validation_data['last']

scaled_features_validation = scaler.transform(features_validation)

validation_data_processed = pd.DataFrame(scaled_features_validation, columns=features_validation.columns)
validation_data_processed['last'] = target_validation.values
validation_data_processed['bid'] = validation_data['bid'].values
validation_data_processed['ask'] = validation_data['ask'].values

# Preprocess the test data
test_data = test_data.drop(columns=['contractID', 'symbol', 'expiration', 'strike', 'date'])
features_test = test_data.drop(columns=['last', 'bid', 'ask'])
target_test = test_data['last']

scaled_features_test = scaler.transform(features_test)

test_data_processed = pd.DataFrame(scaled_features_test, columns=features_test.columns)
test_data_processed['last'] = target_test.values
test_data_processed['bid'] = test_data['bid'].values
test_data_processed['ask'] = test_data['ask'].values

# Define the environment with proportional transaction cost based on bid-ask spread
class OptionHedgingEnv(gym.Env):
    def __init__(self, data, xi=0.1, transaction_cost_proportion=0.05):
        super(OptionHedgingEnv, self).__init__()
        self.data = data.reset_index(drop=True)
        self.current_step = 0
        self.action_space = gym.spaces.Discrete(3)  # Buy, Hold, Sell
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(data.shape[1] - 3,), dtype=np.float32)
        self.position = 0  # Initial position: 0 (no position)
        self.xi = xi  # Risk aversion parameter
        self.transaction_cost_proportion = transaction_cost_proportion  # Proportional transaction cost
        self.wealth = 0.0  # Initialize wealth

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

        # Initialize P&L and Transaction Cost
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

        self.wealth += pnl  # Update wealth

        # Calculate reward based on P&L minus penalty
        reward = pnl - self.xi * abs(pnl)

        done = self.current_step == len(self.data) - 1
        return obs, reward, done, {}


# Define the DQN Model with TensorFlow
def build_dqn_model(input_shape, output_shape):
    model = models.Sequential()
    model.add(layers.Dense(16, input_shape=input_shape, activation='relu'))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(output_shape, activation='linear'))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='mse')
    return model

# Experience Replay Memory
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

# Create the environments with a proportional transaction cost of 0.05
train_env = OptionHedgingEnv(train_data_processed, xi=0.1, transaction_cost_proportion=0.05)
validation_env = OptionHedgingEnv(validation_data_processed, xi=0.1, transaction_cost_proportion=0.05)
test_env = OptionHedgingEnv(test_data_processed, xi=0.1, transaction_cost_proportion=0.05)


input_shape = (train_env.observation_space.shape[0],)
output_shape = train_env.action_space.n
model = build_dqn_model(input_shape, output_shape)

# Replay Memory
memory = ReplayMemory(20)

# Training Parameters
gamma = 0.95
epsilon = 1.0
epsilon_min = 0.1
epsilon_decay = 0.995
batch_size = 64
num_episodes = 5

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

# Training loop with validation
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

# Plot Validation Rewards and Errors
plt.figure(figsize=(12, 6))
plt.plot(validation_rewards, label='Cumulative Validation Rewards', color='blue')
plt.xlabel('Steps')
plt.ylabel('Cumulative Reward')
plt.title('Cumulative Reward over Validation Steps')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(actual_val_rewards, label='Actual Validation Rewards', color='green')
plt.plot(predicted_val_rewards, label='Predicted Validation Rewards', color='red', linestyle='dashed')
plt.xlabel('Steps')
plt.ylabel('Reward')
plt.title('Actual vs Predicted Validation Rewards')
plt.legend()
plt.show()

# Plot Testing Rewards and Errors
plt.figure(figsize=(12, 6))
plt.plot(cumulative_test_rewards, label='Cumulative Testing Rewards', color='blue')
plt.xlabel('Steps')
plt.ylabel('Cumulative Reward')
plt.title('Cumulative Reward over Testing Steps')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(actual_test_rewards, label='Actual Testing Rewards', color='green')
plt.plot(predicted_test_rewards, label='Predicted Testing Rewards', color='red', linestyle='dashed')
plt.xlabel('Steps')
plt.ylabel('Reward')
plt.title('Actual vs Predicted Testing Rewards')
plt.legend()
plt.show()

print(f"Total Test Set Reward: {total_reward_test}")
