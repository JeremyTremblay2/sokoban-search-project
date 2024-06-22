import gym
import gym_sokoban
import numpy as np
from functions import *

env = gym.make('Sokoban-v0')
observation = env.reset()
seed_everything(42, env)

q_table = {}

alpha = 0.5
gamma = 0.95
epsilon = 0.1
num_episodes = 2000

reward_box_placed = 10  # Récompense pour placer une boîte sur un emplacement cible
reward_box_moved = -1    # Récompense pour déplacer une boîte loin d'un emplacement cible
reward_default = -0.1    # Récompense par défaut pour les actions autres que placer ou déplacer des boîtes

rewards_per_episode = []
boxes_placed_per_episode = []
boxes_moved_per_episode = []
waiting_moves_per_episode = []
games_lost_per_episode = []
episode_at_first_win = None

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
        given_reward = reward_default

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
            given_reward = reward_box_placed
            print(f"A box ({info}) has been put on an emplacement: {given_reward}")
        elif game_reward == -1.1:
            boxes_moved += 1
            given_reward = reward_box_moved
            print(f"A box ({info}) has been put away from an emplacement: {given_reward}")
        elif game_reward == -0.1:
            given_reward = reward_default
        elif game_reward > 2:
            is_current_game_lost = 0
            if episode_at_first_win is None:
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

save_and_plot_results(__file__, rewards_per_episode, boxes_placed_per_episode, boxes_moved_per_episode, waiting_moves_per_episode, games_lost_per_episode, episode_at_first_win, num_episodes, q_table)
