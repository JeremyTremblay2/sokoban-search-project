import gym_sokoban
import cv2
import random
import numpy as np
import gym

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    env.seed(seed)

env = gym.make('Sokoban-v0')

seed_everything(42)
observation = env.reset()
print(observation.shape)

done = False
while not done:
    img = env.render(mode='rgb_array')
    cv2.imshow('Sokoban', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.waitKey(200) 

    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    print(f"Action: {action}, Reward: {reward}, Done: {done}, Info: {info}")

cv2.destroyAllWindows() 