# Define the DQN Model with TensorFlow
def build_dqn_model(input_shape, output_shape):
    model = models.Sequential()
    model.add(layers.Dense(64, input_shape=input_shape, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(output_shape, activation='linear'))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='mse')
    return model
