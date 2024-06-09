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

#shape = (observation.shape[0] // 16) * (observation.shape[1] // 16)
q_table = {}

alpha = 0.6  # Augmenter alpha pour accélérer l'apprentissage
gamma = 0.99
epsilon = 0.1  # Ajuster epsilon pour l'exploration/exploitation
num_episodes = 2000  # Augmenter le nombre d'épisodes d'entraînement

reward_box_placed = 10  # Récompense pour placer une boîte sur un emplacement cible
reward_box_moved = -1    # Récompense pour déplacer une boîte loin d'un emplacement cible
reward_default = -0.1    # Récompense par défaut pour les actions autres que placer ou déplacer des boîtes

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
        index = hash(int(''.join(map(str, env.env.env.room_state.flatten()))))

        if not index in q_table:
            q_table[index] = np.zeros(env.action_space.n)

        if np.all(q_table[index] == q_table[index][0]):
            action = env.action_space.sample()
            # print(f"New state: {index}")

        elif random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[index])
            # print(f"Action took: {action}")
            
        if action == 0:
            waiting_moves += 1

        next_state, reward, done, info = env.env.env.step(action, observation_mode='tiny_rgb_array')
        futur_index = hash(int(''.join(map(str, env.env.env.room_state.flatten()))))
        if not futur_index in q_table:
            q_table[futur_index] = np.zeros(env.action_space.n)

        if reward == 0.9:
            reward = reward_box_placed
            boxes_placed += 1
            print(f"A box ({info}) has been put on an emplacement: {reward}")
        elif reward == -1.1:
            reward = reward_box_moved
            boxes_moved += 1
            print(f"A box ({info}) has been put away from an emplacement: {reward}")
        elif reward == -0.1:
            reward = reward_default
        elif reward >= 2:
            print("Game won: ", reward)

        total_reward += reward

        q_table[index][action] += alpha * (reward + gamma * np.max(q_table[futur_index]) - q_table[index][action])
        state = next_state

    rewards_per_episode.append(total_reward)
    boxes_moved_per_episode.append(boxes_moved)
    boxes_placed_per_episode.append(boxes_placed)
    waiting_moves_per_episode.append(waiting_moves)

    if i_episode % 10 == 0 and i_episode != 0:
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
