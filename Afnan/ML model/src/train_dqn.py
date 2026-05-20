import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# Load data
X = np.load("../extracted_features/features.npy").astype(np.float32)
y = np.load("../extracted_features/labels.npy")

num_actions = 6

# Simple model
model = tf.keras.Sequential([
    layers.Input(shape=(1280,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_actions, activation='linear')
])

model.compile(optimizer='adam', loss='mse')

epsilon = 1.0

print("Training started...")

for epoch in range(3):
    print("Epoch", epoch)

    for i in range(len(X)):

        state = X[i].reshape(1, -1)  # reshape for model input

        # choose action
        if np.random.rand() < epsilon:
            action = np.random.randint(num_actions)
        else:
            q_values = model(state, training=False).numpy()
            action = np.argmax(q_values)

        # reward
        reward = 1 if action == y[i] else -1

        # target Q update (simple version)
        target = model(state, training=False).numpy()
        target[0][action] = reward

        # train
        model.train_on_batch(state, target)

    epsilon *= 0.9
    print("epsilon:", epsilon)

model.save("../models/dqn_model.h5")

print("Done ✔")