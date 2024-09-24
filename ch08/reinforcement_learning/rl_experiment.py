import numpy as np
import tensorflow as tf
from scipy.spatial.distance import euclidean
from tensorflow.keras import Model, Sequential, layers, optimizers, metrics, losses
import pandas as pd
from sklearn.preprocessing import StandardScaler
import copy

from environment import Environment
from models import RLModel

state_size = 5
n_actions = 4
epsilon = 1.0
history = {"state": [], "reward": []}
model = RLModel(state_size, n_actions)
n_samples = 1000
max_steps = 500
regrets = []

for i in range(100):
    env = Environment(env_size, max_steps=max_steps)
    while not env.is_done:
        state = env.get_state()
        if np.random.rand() < epsilon:
            action = np.random.randint(n_actions)
        else:
            action = model.predict(state)
        # print(action)
        # print(env.delta)
        reward = env.update(action)
        history["state"].append(np.concatenate([state, [action]]))
        history["reward"].append(reward)
    print(f"Completed episode {i} in {env.total_steps} steps. Ideal steps: {env.ideal_steps}. Epsilon: {epsilon}")
    regrets.append(np.abs(env.total_steps-env.ideal_steps))
    idxs = np.random.choice(len(history["state"]), n_samples)
    model.fit(np.array(history["state"])[idxs], np.array(history["reward"])[idxs])
    epsilon-=epsilon/10
