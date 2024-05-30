import gym
import gym_sokoban
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tensorflow.keras import layers, models

def seed_everything(seed, env):
    random.seed(seed)
    np.random.seed(seed)
    env.seed(seed)

def create_model(input_shape, num_actions):
    model = models.Sequential([
        layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_actions)
    ])
    return model

env = gym.make('Sokoban-v0')
observation = env.reset()
seed_everything(42, env)

alpha = 0.5
gamma = 0.95
epsilon = 0.1
num_episodes = 200

rewards_per_episode = []
boxes_placed_per_episode = []
boxes_moved_per_episode = []

input_shape = (10, 10)
num_actions = env.action_space.n
model = create_model(input_shape, num_actions)
optimizer = tf.keras.optimizers.Adam(learning_rate=alpha)
loss_fn = tf.keras.losses.MeanSquaredError()

for i_episode in range(num_episodes):
    seed_everything(42, env)
    state = env.env.env.reset(render_mode='tiny_rgb_array')
    #print(env.)
    # print(state)
    done = False
    print("-------------------------------")

    total_reward = 0
    boxes_placed = 0
    boxes_moved = 0

    while not done:
        q_values = model.predict(np.expand_dims(state, axis=0), verbose=0)[0]

        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_values)

        next_state, reward, done, info = env.env.env.step(action, observation_mode='tiny_rgb_array')
        # print(next_state.__dict__)

        if reward == 0.9:
            boxes_placed += 1
            print(f"A box ({info}) has been put on an emplacement: {reward}")
        elif reward == -1.1:
            boxes_moved += 1
            print(f"A box ({info}) has been put away from an emplacement: {reward}")
        elif reward == -0.1:
            reward = 0
        elif reward == 9.9:
            print("Game won: ", reward)

        if done:
            target = reward
        else:
            next_q_values = model.predict(np.expand_dims(next_state, axis=0), verbose=0)[0]
            target = reward + gamma * np.max(next_q_values)

        q_values[action] = target
        with tf.GradientTape() as tape:
            predictions = model(np.expand_dims(state, axis=0))
            loss = loss_fn(q_values, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        state = next_state
        total_reward += reward
    
    rewards_per_episode.append(total_reward)
    boxes_moved_per_episode.append(boxes_moved)
    boxes_placed_per_episode.append(boxes_placed)

    if i_episode % 10 == 0 and i_episode != 0:
        print(f"Current episode: {i_episode}")

print("Training finished.\n")

plt.plot(rewards_per_episode)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Rewards per Episode')
plt.grid(True)
plt.show()

plt.plot(boxes_placed_per_episode)
plt.xlabel('Episode')
plt.ylabel('Boxes Placed on Target')
plt.title('Boxes Placed on Target per Episode')
plt.grid(True)
plt.show()

plt.plot(boxes_moved_per_episode)
plt.xlabel('Episode')
plt.ylabel('Boxes Moved Away from Target')
plt.title('Boxes Moved Away from Target per Episode')
plt.grid(True)
plt.show()

model.save("sokoban_dql_model.h5")