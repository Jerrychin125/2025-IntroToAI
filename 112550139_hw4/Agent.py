import numpy as np

class Agent:
    def __init__(self, k: int, epsilon: float,
                 alpha: float | None = None, seed: int | None = None):
        self.k = k
        self.epsilon = epsilon  # Exploration rate
        self.alpha = alpha      # Learning rate
        self.random_state = np.random.RandomState(seed)
        self.reset()

    # ---------- interaction ----------
    def select_action(self) -> int:
        if self.random_state.rand() < self.epsilon:
            # Exploration: choose a random action
            return self.random_state.randint(self.k)
        # Exploitation: choose the best action
        max_q = np.max(self.Q)
        best_actions = np.flatnonzero(self.Q == max_q)
        return int(self.random_state.choice(best_actions))

    def update_q(self, action: int, reward: float):
        self.N[action] += 1
        if self.alpha is None: # sample-average         
            step_size = 1.0 / self.N[action]
        else: # constant learning rate
            step_size = self.alpha
        # Q(a) ← Q(a) + α * (R - Q(a))
        self.Q[action] += step_size * (reward - self.Q[action])

    def reset(self):
        self.Q = np.zeros(self.k, dtype=np.float64)
        self.N = np.zeros(self.k, dtype=np.int64)
