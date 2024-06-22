import gym
import gym_sokoban
import numpy as np
import optuna
from functions import *
from multiprocessing import cpu_count

def train_agent(trial):
    alpha = trial.suggest_float('alpha', 0.1, 0.9, step=0.2)
    gamma = trial.suggest_float('gamma', 0.8, 0.99, step=0.05)
    epsilon = trial.suggest_float('epsilon', 0.1, 0.5, step=0.1)
    num_episodes = 1000  

    env = gym.make('Sokoban-v0')
    seed_everything(42, env)

    q_table = {}
    rewards_per_episode = []
    episode_at_first_win = None
    is_current_game_lost = 1

    for i_episode in range(num_episodes):
        seed_everything(42, env)
        state = env.reset(render_mode='tiny_rgb_array')
        done = False
        total_reward = 0
        is_current_game_lost = 1

        while not done:
            index = hash(int(''.join(map(str, env.env.env.room_state.flatten()))))
            given_reward = -0.1

            if index not in q_table:
                q_table[index] = np.zeros(env.action_space.n - 1)

            if np.all(q_table[index] == q_table[index][0]):
                action = env.action_space.sample() - 1
                while action == -1:
                    action = env.action_space.sample() - 1
            elif np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample() - 1
                while action == -1:
                    action = env.action_space.sample() - 1
            else:
                action = np.argmax(q_table[index])

            next_state, game_reward, done, info = env.env.env.step(action, observation_mode='tiny_rgb_array')

            futur_index = hash(int(''.join(map(str, env.env.env.room_state.flatten()))))
            if futur_index not in q_table:
                q_table[futur_index] = np.zeros(env.action_space.n - 1)

            if game_reward == 0.9:
                given_reward += 1
            elif game_reward == -1.1:
                given_reward += -1
            elif game_reward > 2:
                is_current_game_lost = 0
                if episode_at_first_win is None:
                    episode_at_first_win = i_episode
                given_reward += 10
                done = True

            if determine_direction_based_on_action(action) in [PUSH_UP, PUSH_DOWN, PUSH_LEFT, PUSH_RIGHT] and is_game_lost(env.env.env.room_state):
                given_reward += -10
                is_current_game_lost = 2
                done = True

            total_reward += given_reward
            q_table[index][action] += alpha * (given_reward + gamma * np.max(q_table[futur_index]) - q_table[index][action])
            state = next_state

        rewards_per_episode.append(total_reward)

        if i_episode % 10 == 0 and i_episode != 0:
            print(f"Episode {i_episode} - Average reward: {np.mean(rewards_per_episode)}")

        # Condition d'arrêt si l'agent a réussi à finir le niveau une fois
        if is_current_game_lost == 0:
            break

    return np.mean(rewards_per_episode)

if __name__ == '__main__':
    study = optuna.create_study(direction='maximize')
    study.optimize(train_agent, n_trials=cpu_count(), n_jobs=cpu_count())

    best_params = study.best_params
    best_value = study.best_value
    print(f"Best parameters: {best_params} with average reward {best_value}")

    with open('results/best_hyper_parameters.txt', 'w') as f:
        f.write(f"Best parameters: {best_params} with average reward {best_value}")