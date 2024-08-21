# final DDPG
import gym
import numpy as np
import pandas as pd
import random
import tensorflow as tf
from tensorflow.keras import models, layers
from collections import deque
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

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
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)  # Continuous action space
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

        if action < -0.5:  # Buy
            transaction_cost = self.transaction_cost_proportion * bid_ask_spread
            pnl = curr_price - curr_ask_price - transaction_cost  # Only consider last price for P&L
            self.position = 1

        elif action > 0.5:  # Sell
            transaction_cost = self.transaction_cost_proportion * bid_ask_spread
            pnl = curr_bid_price - curr_price - transaction_cost  # Only consider last price for P&L
            self.position = -1

        elif -0.5 <= action <= 0.5:  # Hold
            if self.position == 1:
                pnl = curr_price - prev_price  # Only consider last price for P&L
            elif self.position == -1:
                pnl = prev_price - curr_price  # Only consider last price for P&L

        self.wealth += pnl  # Update wealth

        # Calculate reward based on P&L minus penalty
        reward = pnl - self.xi * abs(pnl)

        done = self.current_step == len(self.data) - 1
        return obs, reward, done, {}


# Define the Actor-Critic Models with TensorFlow
def build_actor(input_shape, action_dim):
    model = models.Sequential()
    model.add(layers.Dense(16, input_shape=input_shape, activation='relu'))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(action_dim, activation='tanh'))  # Tanh for continuous actions
    return model

def build_critic(input_shape, action_dim):
    state_input = layers.Input(shape=input_shape)
    action_input = layers.Input(shape=(action_dim,))
    concat = layers.Concatenate()([state_input, action_input])
    x = layers.Dense(16, activation='relu')(concat)
    x = layers.Dense(16, activation='relu')(x)
    output = layers.Dense(1)(x)
    return models.Model(inputs=[state_input, action_input], outputs=output)

# Hyperparameters and Initialization
gamma = 0.95
tau = 0.01  # Target network update parameter
actor_lr = 1e-3
critic_lr = 1e-3
buffer_size = 20
batch_size = 64
num_episodes = 5

# Environment and Models Initialization
train_env = OptionHedgingEnv(train_data_processed, xi=0.1, transaction_cost_proportion=0.05)
validation_env = OptionHedgingEnv(validation_data_processed, xi=0.1, transaction_cost_proportion=0.05)
test_env = OptionHedgingEnv(test_data_processed, xi=0.1, transaction_cost_proportion=0.05)

input_shape = (train_env.observation_space.shape[0],)
action_dim = train_env.action_space.shape[0]

# Actor-Critic Models
actor = build_actor(input_shape, action_dim)
critic = build_critic(input_shape, action_dim)
target_actor = build_actor(input_shape, action_dim)
target_critic = build_critic(input_shape, action_dim)

# Optimizers
actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)
critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)

# Experience Replay Memory
memory = deque(maxlen=buffer_size)

# Target Network Initialization
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

# Function to select an action using the actor network with added noise for exploration
def select_action(state):
    state = np.expand_dims(state, axis=0)
    action = actor(state, training=False).numpy()[0]
    noise = np.random.normal(scale=0.05, size=action_dim)  # Exploration noise
    return np.clip(action + noise, train_env.action_space.low, train_env.action_space.high)

# Function to train the models
def train():
    if len(memory) < batch_size:
        return
    mini_batch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*mini_batch)

    states = np.array(states, dtype=np.float32)
    actions = np.array(actions, dtype=np.float32)
    rewards = np.array(rewards, dtype=np.float32).reshape(-1, 1)
    next_states = np.array(next_states, dtype=np.float32)
    dones = np.array(dones, dtype=np.float32).reshape(-1, 1)

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

validation_cumulative_rewards = []  # Store cumulative rewards at each time step during validation
testing_cumulative_rewards = []     # Store cumulative rewards at each time step during testing

for episode in range(num_episodes):
    # Training phase
    state = train_env.reset()
    total_reward_train = 0
    while True:
        action = select_action(state)
        next_state, reward, done, _ = train_env.step(action)
        memory.append((state, action, reward, next_state, done))
        state = next_state
        total_reward_train += reward
        train()
        if done:
            break

    # Validation phase
    state_val = validation_env.reset()
    cumulative_reward_val = 0  # Reset cumulative reward for validation
    while True:
        action_val = select_action(state_val)
        next_state_val, reward_val, done_val, _ = validation_env.step(action_val)
        cumulative_reward_val += reward_val  # Accumulate reward at each time step
        validation_cumulative_rewards.append(cumulative_reward_val)  # Store cumulative reward at each time step
        state_val = next_state_val
        if done_val:
            break

    print(f"Episode {episode + 1}, Training Reward: {total_reward_train}, Validation Cumulative Reward: {cumulative_reward_val}")

# Testing phase
for _ in range(num_episodes):  # Run through the number of episodes to test
    state_test = test_env.reset()
    cumulative_reward_test = 0  # Reset cumulative reward for testing
    while True:
        action_test = select_action(state_test)
        next_state_test, reward_test, done_test, _ = test_env.step(action_test)
        cumulative_reward_test += reward_test  # Accumulate reward at each time step
        testing_cumulative_rewards.append(cumulative_reward_test)  # Store cumulative reward at each time step
        state_test = next_state_test
        if done_test:
            break

# Plot Validation Rewards by Time Steps
plt.figure(figsize=(12, 6))
plt.plot(validation_cumulative_rewards, label='Cumulative Validation Rewards', color='blue')
plt.xlabel('Time Steps')
plt.ylabel('Cumulative Reward')
plt.title('Cumulative Reward over Time Steps during Validation')
plt.legend()
plt.show()

# Plot Testing Rewards by Time Steps
plt.figure(figsize=(12, 6))
plt.plot(testing_cumulative_rewards, label='Cumulative Testing Rewards', color='blue')
plt.xlabel('Time Steps')
plt.ylabel('Cumulative Reward')
plt.title('Cumulative Reward over Time Steps during Testing')
plt.legend()
plt.show()

# Print the final cumulative reward after all episodes
print(f"Final Cumulative Test Reward: {testing_cumulative_rewards[-1]}")
