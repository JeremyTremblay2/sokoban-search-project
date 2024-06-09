import gym
import gym_sokoban
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
import time

def seed_everything(seed, env):
    random.seed(seed)
    np.random.seed(seed)
    env.seed(seed)

env = gym.make('Sokoban-v0')
observation = env.reset()
seed_everything(42, env)

#shape = (observation.shape[0] // 16) * (observation.shape[1] // 16)
q_table = {}

alpha = 0.5
gamma = 0.95
epsilon = 0.1
num_episodes = 1000

rewards_per_episode = []
boxes_placed_per_episode = []
boxes_moved_per_episode = []
waiting_moves_per_episode = []

def can_push_box(board, r, c, dr, dc):
    """
    Vérifie si une caisse à la position (r, c) peut être poussée dans la direction (dr, dc)
    vers une nouvelle case marchable.
    """
    rows, cols = board.shape
    r_next, c_next = r + dr, c + dc
    r_prev, c_prev = r - dr, c - dc

    if 0 <= r_next < rows and 0 <= c_next < cols and 0 <= r_prev < rows and 0 <= c_prev < cols:
        if board[r][c] in {3, 4} and board[r_next][c_next] in {1, 2, 5, 6} and board[r_prev][c_prev] in {1, 2, 5, 6}:
            return True
    return False
    
def is_game_lost(board):
    width = len(board)
    cols = len(board[0])
    for r in range(width):
        for c in range(cols):
            if board[r][c] == 4:
                if ((board[r-1][c] == 0 and board[r][c-1] == 0) or
                    (board[r-1][c] == 0 and board[r][c+1] == 0) or
                    (board[r+1][c] == 0 and board[r][c-1] == 0) or
                    (board[r+1][c] == 0 and board[r][c+1] == 0)):
                    return True
    return False

for i_episode in range(num_episodes):
    seed_everything(42, env)
    state = env.reset(render_mode='tiny_rgb_array')
    done = False
    print("-------------------------------")

    total_reward = 0
    boxes_placed = 0
    boxes_moved = 0
    wainting_moves = 0
    number_of_moves = 0

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
            wainting_moves += 1

        next_state, reward, done, info = env.env.env.step(action, observation_mode='tiny_rgb_array')
        number_of_moves += 1
            
        futur_index = hash(int(''.join(map(str, env.env.env.room_state.flatten()))))
        if not futur_index in q_table:
            q_table[futur_index] = np.zeros(env.action_space.n)

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
            reward = 10

        if (is_game_lost(env.env.env.room_state)):
            print(env.env.env.room_state)
            print(f"Game lost: {reward} after {number_of_moves}")
            reward = -10
            done = 10

        total_reward += reward

        q_table[index][action] += alpha * (reward + gamma * np.max(q_table[futur_index]) - q_table[index][action])
        state = next_state

    rewards_per_episode.append(total_reward)
    boxes_moved_per_episode.append(boxes_moved)
    boxes_placed_per_episode.append(boxes_placed)
    waiting_moves_per_episode.append(wainting_moves)

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
plt.ylabel('Total wainting moves')
plt.title('Wainting Moves per Episode (0)')
plt.grid(True)

plt.tight_layout()
plt.show()

print("Training finished.\n")
print(q_table)

print("Saving results...\n")
with open("q_table.txt", "w") as f:
    f.write(str(q_table))