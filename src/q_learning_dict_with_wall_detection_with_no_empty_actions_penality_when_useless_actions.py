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
num_episodes = 20

rewards_per_episode = []
boxes_placed_per_episode = []
boxes_moved_per_episode = []
waiting_moves_per_episode = []
games_lost_per_episode = []
episode_at_first_win = None

UP = 'up'
DOWN = 'down'
LEFT = 'left'
RIGHT = 'right'
PUSH_UP = 'push_up'
PUSH_DOWN = 'push_down'
PUSH_LEFT = 'push_left'
PUSH_RIGHT = 'push_right'

def determine_agent_position(board):
    for i, room in enumerate(board):
        for j, cell in enumerate(room):
            if cell in [5, 6]:
                return i, j
            
def determine_direction_based_on_action(action):
    if action == 0:
        return UP
    elif action == 1:
        return DOWN
    elif action == 2:
        return LEFT
    elif action == 3:
        return RIGHT
    elif action == 4:
        return PUSH_UP
    elif action == 5:
        return PUSH_DOWN
    elif action == 6:
        return PUSH_LEFT
    elif action == 7:
        return PUSH_RIGHT
    else:
        return None

def can_push_box(board, x, y, direction):
    rows, cols = board.shape
    if direction == PUSH_UP:
        if x - 1 >= 0 and board[x - 1][y] in [1, 2, 5, 6] and board[x - 2][y] in [1, 2, 5, 6]:
            return True
    elif direction == PUSH_DOWN:
        if x + 1 < rows and board[x + 1][y] in [1, 2, 5, 6] and board[x + 2][y] in [1, 2, 5, 6]:
            return True
    elif direction == PUSH_LEFT:
        if y - 1 >= 0 and board[x][y - 1] in [1, 2, 5, 6] and board[x][y - 2] in [1, 2, 5, 6]:
            return True
    elif direction == PUSH_RIGHT:
        if y + 1 < cols and board[x][y + 1] in [1, 2, 5, 6] and board[x][y + 2] in [1, 2, 5, 6]:
            return True
    return False

def can_agent_move(board, x, y, direction):
    rows, cols = board.shape
    if direction == UP:
        if x - 1 >= 0 and board[x - 1][y] in [1, 2, 5, 6]:
            return True
    elif direction == DOWN:
        if x + 1 < rows and board[x + 1][y] in [1, 2, 5, 6]:
            return True
    elif direction == LEFT:
        if y - 1 >= 0 and board[x][y - 1] in [1, 2, 5, 6]:
            return True
    elif direction == RIGHT:
        if y + 1 < cols and board[x][y + 1] in [1, 2, 5, 6]:
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

    total_reward = 0
    boxes_placed = 0
    boxes_moved = 0
    wainting_moves = 0
    is_current_game_lost = 1
    number_of_moves = 0

    while not done:
        index = hash(int(''.join(map(str, env.env.env.room_state.flatten()))))
        given_reward = 0

        if not index in q_table:
            q_table[index] = np.zeros(env.action_space.n - 1)

        if np.all(q_table[index] == q_table[index][0]):
            action = env.action_space.sample() - 1
            while (action == -1):
                action = env.action_space.sample() - 1

        elif random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() - 1
            while (action == -1):
                action = env.action_space.sample() - 1
        else:
            action = np.argmax(q_table[index])
            agent_index = determine_agent_position(env.env.env.room_state)
            direction = determine_direction_based_on_action(action)
            if direction in [PUSH_UP, PUSH_DOWN, PUSH_LEFT, PUSH_RIGHT] and not can_push_box(env.env.env.room_state, agent_index[0], agent_index[1], direction):
                given_reward = -0.1
            elif direction in [UP, DOWN, LEFT, RIGHT] and not can_agent_move(env.env.env.room_state, agent_index[0], agent_index[1], direction):
                given_reward = -0.1
            # print(f"Action took: {action}")
            
        if action == -1:
            wainting_moves += 1

        next_state, game_reward, done, info = env.env.env.step(action, observation_mode='tiny_rgb_array')
        number_of_moves += 1
            
        futur_index = hash(int(''.join(map(str, env.env.env.room_state.flatten()))))
        if not futur_index in q_table:
            q_table[futur_index] = np.zeros(env.action_space.n - 1)

        if game_reward == 0.9:
            boxes_placed += 1
            given_reward = 1
            print(f"A box ({info}) has been put on an emplacement: {given_reward}")
        elif game_reward == -1.1:
            boxes_moved += 1
            given_reward = -1
            print(f"A box ({info}) has been put away from an emplacement: {given_reward}")
        elif game_reward == -0.1:
            pass
        elif game_reward == 9.9:
            is_current_game_lost = 0
            episode_at_first_win = i_episode
            print(f"Game won: {given_reward} after {number_of_moves} moves")
            given_reward = 10

        if (determine_direction_based_on_action(action) in [PUSH_UP, PUSH_DOWN, PUSH_LEFT, PUSH_RIGHT] and is_game_lost(env.env.env.room_state)):
            print(env.env.env.room_state)
            given_reward = -10
            is_current_game_lost = 2
            print(f"Game lost: {given_reward} after {number_of_moves} moves")
            done = True

        total_reward += given_reward

        q_table[index][action] += alpha * (given_reward + gamma * np.max(q_table[futur_index]) - q_table[index][action])
        state = next_state

    rewards_per_episode.append(total_reward)
    boxes_moved_per_episode.append(boxes_moved)
    boxes_placed_per_episode.append(boxes_placed)
    waiting_moves_per_episode.append(wainting_moves)
    games_lost_per_episode.append(is_current_game_lost)

    if i_episode % 10 == 0 and i_episode != 0:
        print(f"Current episode: {i_episode}")

print(f"Episode at first win: {episode_at_first_win}.")

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
plt.plot(games_lost_per_episode)
plt.xlabel('Episode')
plt.ylabel('Status (0: No, 1: Yes, 2: Yes, game lost)')
plt.title('Was the game lost?')
plt.grid(True)

plt.tight_layout()
plt.show()

print("Training finished.\n")
print(q_table)

print("Saving results...\n")
with open("q_table.txt", "w") as f:
    f.write(str(q_table))