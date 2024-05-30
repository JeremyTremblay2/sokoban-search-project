import gym
import gym_sokoban
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt

def seed_everything(seed, env):
    random.seed(seed)
    np.random.seed(seed)
    env.seed(seed)

env = gym.make('Sokoban-v0')
observation = env.reset()
seed_everything(42, env)

# shape = (observation.shape[0] // 16) * (observation.shape[1] // 16)
q_table = {}

alpha = 0.1
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
num_episodes = 5000

rewards_per_episode = []
boxes_placed_per_episode = []
boxes_moved_per_episode = []
waiting_moves_per_episode = []

for i_episode in range(num_episodes):
    seed_everything(42, env)
    state = env.reset(render_mode='tiny_rgb_array')
    done = False
    print("-------------------------------")

    total_reward = 0
    boxes_placed = 0
    boxes_moved = 0
    waiting_moves = 0

    while not done:
        index = hash(tuple(env.env.env.room_state.flatten()))

        if index not in q_table:
            q_table[index] = np.zeros(env.action_space.n)

        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[index])

        if action == 0:
            waiting_moves += 1

        next_state, reward, done, info = env.env.env.step(action, observation_mode='tiny_rgb_array')
        future_index = hash(tuple(env.env.env.room_state.flatten()))

        if future_index not in q_table:
            q_table[future_index] = np.zeros(env.action_space.n)

        if reward == 0.9:
            boxes_placed += 1
            print(f"A box ({info}) has been put on an emplacement: {reward}")
        elif reward == -1.1:
            boxes_moved += 1
            print(f"A box ({info}) has been put away from an emplacement: {reward}")
        elif reward == -0.1:
            reward = -0.1
        elif reward >= 2:
            print("Game won: ", reward)

        total_reward += reward

        q_table[index][action] += alpha * (reward + gamma * np.max(q_table[future_index]) - q_table[index][action])
        state = next_state

    rewards_per_episode.append(total_reward)
    boxes_moved_per_episode.append(boxes_moved)
    boxes_placed_per_episode.append(boxes_placed)
    waiting_moves_per_episode.append(waiting_moves)

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    if i_episode % 100 == 0 and i_episode != 0:
        print(f"Current episode: {i_episode}")

plt.figure(figsize=(10, 10))

plt.subplot(2, 2, 1)
plt.plot(rewards_per_episode)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Rewards per Episode')
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(boxes_placed_per_episode)
plt.xlabel('Episode')
plt.ylabel('Boxes Placed on Target')
plt.title('Boxes Placed on Target per Episode')
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(boxes_moved_per_episode)
plt.xlabel('Episode')
plt.ylabel('Boxes Moved Away from Target')
plt.title('Boxes Moved Away from Target per Episode')
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(waiting_moves_per_episode)
plt.xlabel('Episode')
plt.ylabel('Total Waiting Moves')
plt.title('Waiting Moves per Episode (0)')
plt.grid(True)

plt.tight_layout()
plt.show()

print("Training finished.\n")
print(q_table)

print("Saving results...\n")
with open("q_table.txt", "w") as f:
    f.write(str(q_table))
