import os
import pickle
import csv
import random
import numpy as np
import matplotlib.pyplot as plt

def seed_everything(seed, env):
    random.seed(seed)
    np.random.seed(seed)
    env.seed(seed)

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

def save_and_plot_results(filename, rewards_per_episode, boxes_placed_per_episode, boxes_moved_per_episode, waiting_moves_per_episode, games_lost_per_episode, episode_at_first_win, num_episodes, q_table):
    print(f"Episode at first win: {episode_at_first_win}.")

    fig = plt.figure(figsize=(10, 10))

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
    print("Saving results...\n")

    results_dir = f"results/{os.path.splitext(os.path.basename(filename))[0]}"
    os.makedirs(results_dir, exist_ok=True)

    with open(f"{results_dir}/q_table.bin", "wb") as f:
        pickle.dump(q_table, f)

    variables = {
        "rewards_per_episode": rewards_per_episode,
        "boxes_placed_per_episode": boxes_placed_per_episode,
        "boxes_moved_per_episode": boxes_moved_per_episode,
        "waiting_moves_per_episode": waiting_moves_per_episode,
        "games_lost_per_episode": games_lost_per_episode,
        "episode_at_first_win": episode_at_first_win
    }

    with open(f"{results_dir}/variables.bin", "wb") as f:
        pickle.dump(variables, f)

    with open(f"{results_dir}/results.csv", mode='w') as file:
        writer = csv.writer(file)
        writer.writerow(["Episode", "Total Reward", "Boxes Placed on Target", "Boxes Moved Away from Target", "Total Waiting Moves", "Game Lost"])
        for i in range(num_episodes):
            writer.writerow([i, rewards_per_episode[i], boxes_placed_per_episode[i], boxes_moved_per_episode[i], waiting_moves_per_episode[i], games_lost_per_episode[i]])
        writer.writerow(["Episode at first win", episode_at_first_win])

    fig.savefig(f"{results_dir}/figures.png")

    print("Results saved.")