import numpy as np

class BanditEnv:
    def __init__(self, n_arms=10, epsilon=0.1):
        """
        Initialize the Bandit environment with a given number of arms and epsilon value.
        :param n_arms: Number of arms in the bandit environment.
        :param
        :param epsilon: Epsilon value for exploration-exploitation trade-off.
        """
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.q_values = [0.0] * n_arms
        self.best_action = 0
        self.best_value = 0.0

    def reset(self):
        self.q_values = [0.0] * self.n_arms
        self.best_action = 0
        self.best_value = 0.0

    def step(self, action):
        reward = self.q_values[action] + np.random.normal()
        return reward
    
    def get_history(self):
        action_history = [i for i in range(self.n_arms)]
        reward_history = [self.q_values[i] for i in range(self.n_arms)]
        return action_history, reward_history
    