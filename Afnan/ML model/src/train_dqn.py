import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
import joblib
import os

# ==========================
# Load Features
# ==========================

X = np.load(
    "extracted_features/extracted_train/features.npy"
).astype(np.float32)

y = np.load(
    "extracted_features/extracted_train/labels.npy"
)

print("X Shape:", X.shape)
print("Labels:", np.unique(y))

num_actions = len(np.unique(y))

print("Number of classes:", num_actions)

# ==========================
# Normalize Features
# ==========================

scaler = StandardScaler()

X = scaler.fit_transform(X)

os.makedirs("Models", exist_ok=True)

joblib.dump(
    scaler,
    "Models/scaler.pkl"
)

# ==========================
# Shuffle Data
# ==========================

indices = np.arange(len(X))
np.random.shuffle(indices)

X = X[indices]
y = y[indices]

# ==========================
# Build Network
# ==========================

model = tf.keras.Sequential([
    layers.Input(shape=(1280,)),

    layers.Dense(
        256,
        activation='relu'
    ),

    layers.Dense(
        128,
        activation='relu'
    ),

    layers.Dense(
        64,
        activation='relu'
    ),

    layers.Dense(
        num_actions,
        activation='linear'
    )
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss='mse'
)

# ==========================
# Training Parameters
# ==========================

epochs = 200

epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.995

# ==========================
# Training Loop
# ==========================

print("Training Started...\n")

for epoch in range(epochs):

    correct = 0

    for i in range(len(X)):

        state = X[i].reshape(1, -1)

        # ----------------------
        # Epsilon Greedy
        # ----------------------

        if np.random.rand() < epsilon:

            action = np.random.randint(
                num_actions
            )

        else:

            q_values = model(
                state,
                training=False
            ).numpy()

            action = np.argmax(
                q_values
            )

        # ----------------------
        # Reward
        # ----------------------

        reward = 1 if action == y[i] else -1

        if reward == 1:
            correct += 1

        # ----------------------
        # Target Update
        # ----------------------

        target = model(
            state,
            training=False
        ).numpy()

        # Reward correct class
        target[0][y[i]] = 1

        # Punish wrong action
        if action != y[i]:
            target[0][action] = -1

        model.train_on_batch(
            state,
            target
        )

    # ----------------------
    # Decay Exploration
    # ----------------------

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    train_acc = (
        correct / len(X)
    ) * 100

    print(
        f"Epoch {epoch+1}/{epochs} | "
        f"Accuracy: {train_acc:.2f}% | "
        f"Epsilon: {epsilon:.4f}"
    )

# ==========================
# Save Model
# ==========================

model.save(
    "Models/dqn_model.h5"
)

print("\nTraining Complete!")
print("Model Saved: Models/dqn_model.h5")
print("Scaler Saved: Models/scaler.pkl")