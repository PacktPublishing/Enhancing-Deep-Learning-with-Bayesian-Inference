import numpy as np
import tensorflow as tf
from scipy.spatial.distance import euclidean
from tensorflow.keras import Model, Sequential, layers, optimizers, metrics, losses
import pandas as pd
from sklearn.preprocessing import StandardScaler
import copy

class Environment():

    def __init__(self, env_size=8, max_steps=2000):
        self.env_size = env_size
        self.max_steps = max_steps
        self.agent_location = np.zeros(2)
        self.target_location = np.random.randint(0, self.env_size, 2)
        self.action_space = {
                                0: np.array([0, 1]),
                                1: np.array([0, -1]),
                                2: np.array([1, 0]),
                                3: np.array([-1, 0])
                            }
        self.delta = self.compute_distance()
        self.is_done = False
        self.total_steps = 0
        self.ideal_steps = self.calculate_ideal_steps()

    def update(self, action_int):
        self.agent_location = self.agent_location + self.action_space[action_int]
        # prevent the agent from moving outside the bounds of the environment
        self.agent_location[self.agent_location > (self.env_size-1)] = self.env_size - 1
        self.compute_reward()
        self.total_steps += 1
        self.is_done = (self.delta == 0) or (self.total_steps >= self.max_steps)
        return self.reward

    def compute_distance(self):
        return euclidean(self.agent_location, self.target_location)

    def compute_reward(self):
        d1 = self.delta
        self.delta = self.compute_distance()
        if self.delta < d1:
            self.reward = 10
        else:
            self.reward = 1

    def get_state(self):
        return np.concatenate([self.agent_location, self.target_location])

    def calculate_ideal_action(self, agent_location, target_location):
        min_delta = 1e1000
        ideal_action = -1
        for k in self.action_space.keys():
            delta = euclidean(agent_location + self.action_space[k], target_location)
            if delta <= min_delta:
                min_delta = delta
                ideal_action = k
        return ideal_action, min_delta

    def calculate_ideal_steps(self):
        agent_location = copy.deepcopy(self.agent_location)
        target_location = copy.deepcopy(self.target_location)
        delta = 1e1000
        i = 0
        while delta > 0:
            ideal_action, delta = self.calculate_ideal_action(agent_location, target_location)
            agent_location += self.action_space[ideal_action]
            i+=1
        return i


class EnvironmentUncertainty():

    def __init__(self, env_size=8, max_steps=2000, obstacle=False):
        self.env_size = env_size
        self.max_steps = max_steps
        self.agent_location = np.zeros(2)
        self.obstacle = obstacle
        # Ensure there's always some distance between the agent and the target
        self.target_location = np.random.randint(0, self.env_size, 2)
        while euclidean(self.agent_location, self.target_location) < 4:
            self.target_location = np.random.randint(0, self.env_size, 2)
        self.action_space = {
                                0: np.array([0, 1]),
                                1: np.array([0, -1]),
                                2: np.array([1, 0]),
                                3: np.array([-1, 0])
                            }
        self.delta = self.compute_distance()
        self.is_done = False
        self.total_steps = 0
        self.obstacle_location = np.random.randint(0, self.env_size, 2)
        self.ideal_steps = self.calculate_ideal_steps()

    def update(self, action_int):
        self.agent_location = self.agent_location + self.action_space[action_int]
        # prevent the agent from moving outside the bounds of the environment
        self.agent_location[self.agent_location > (self.env_size-1)] = self.env_size - 1
        self.compute_reward()
        self.total_steps += 1
        self.is_done = (self.delta == 0) or (self.total_steps >= self.max_steps)
        if self.obstacle and not self.is_done:
            self.is_done = euclidean(self.agent_location, self.obstacle_location) == 0
        return self.reward

    def compute_distance(self):
        return euclidean(self.agent_location, self.target_location)

    def compute_reward(self):
        d1 = self.delta
        self.delta = self.compute_distance()
        if self.delta < d1:
            self.reward = 10
        else:
            self.reward = 1

    def get_state(self):
        return np.concatenate([self.agent_location, self.target_location])

    def get_obstacle_proximity(self):
        if len(self.obstacle_location) > 0:
            obstacle_action_dists = np.array([euclidean(self.agent_location+self.action_space[k], self.obstacle_location) for k in self.action_space.keys()])
            return np.array(obstacle_action_dists < 2.5, dtype=float)*20
        else:
            return np.zeros(len(self.action_space))

    def calculate_ideal_action(self, agent_location, target_location):
        min_delta = 1e1000
        ideal_action = -1
        for k in self.action_space.keys():
            delta = euclidean(agent_location + self.action_space[k], target_location)
            if delta <= min_delta:
                min_delta = delta
                ideal_action = k
        return ideal_action, min_delta

    def calculate_ideal_steps(self):
        agent_location = copy.deepcopy(self.agent_location)
        target_location = copy.deepcopy(self.target_location)
        delta = 1e1000
        i = 0
        while delta > 0:
            ideal_action, delta = self.calculate_ideal_action(agent_location, target_location)
            agent_location += self.action_space[ideal_action]
            if np.random.randint(0, 2) and len(self.obstacle_location) == 0 and self.obstacle:
                self.obstacle_location = agent_location
            i+=1
        return i
