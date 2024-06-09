import gym
import gym_sokoban
import numpy as np
import random
from functions import *
from skopt import forest_minimize
from joblib import Parallel, delayed

def evaluate_rewards(reward_params):
    reward_placed = reward_params[0]
    reward_moved = reward_params[1]
    reward_won = reward_params[2]
    reward_lost = reward_params[3]

    env = gym.make('Sokoban-v0')
    observation = env.reset()
    seed_everything(42, env)

    q_table = {}
    total_rewards = []
    num_episodes = 500
    alpha = 0.5
    gamma = 0.95
    epsilon = 0.1

    for i_episode in range(num_episodes):
        seed_everything(42, env)
        state = env.reset(render_mode='tiny_rgb_array')
        done = False
        total_reward = 0

        while not done:
            index = hash(int(''.join(map(str, env.env.env.room_state.flatten()))))
            given_reward = -0.1

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
                
            futur_index = hash(int(''.join(map(str, env.env.env.room_state.flatten()))))
            if not futur_index in q_table:
                q_table[futur_index] = np.zeros(env.action_space.n - 1)

            if game_reward == 0.9:
                given_reward += reward_placed
                print(f"A box ({info}) has been put on an emplacement: {given_reward}")
            elif game_reward == -1.1:
                given_reward += -reward_moved
                print(f"A box ({info}) has been put away from an emplacement: {given_reward}")
            elif game_reward == -0.1:
                pass
            elif game_reward > 2:
                print(f"Game won: {given_reward} after moves")
                given_reward += reward_won
                done = True

            if (is_game_lost(env.env.env.room_state)):
                given_reward += reward_lost
                print(f"Game lost: {given_reward} after moves")
                done = True

            total_reward += given_reward

            q_table[index][action] += alpha * (given_reward + gamma * np.max(q_table[futur_index]) - q_table[index][action])
            state = next_state

        if i_episode % 10 == 0 and i_episode != 0:
            print(f"Current episode: {i_episode}")
        
        total_rewards.append(total_reward)
    
    return -np.mean(total_rewards)

env = gym.make('Sokoban-v0')
observation = env.reset()
seed_everything(42, env)

def optimize_rewards():
    space = [
        (-10.0, 10.0),  # reward_placed
        (-10.0, 10.0),  # reward_moved
        (0, 30),  # reward_won
        (-30, 0),   # reward_lost
    ]

    res = forest_minimize(evaluate_rewards, space, n_calls=10, random_state=42, n_jobs=-1)

    print("Meilleures valeurs de récompense :")
    print("reward_placed: ", res.x[0])
    print("reward_moved: ", res.x[1])
    print("reward_won: ", res.x[2])
    print("reward_lost: ", res.x[3])


    with open("results/best_rewards.txt", "w") as f:
        f.write("Meilleures valeurs de récompense :\n")
        f.write(f"reward_placed: {res.x[0]}\n")
        f.write(f"reward_moved: {res.x[1]}\n")
        f.write(f"reward_won: {res.x[2]}\n")
        f.write(f"reward_lost: {res.x[3]}\n")
    
optimize_rewards()

