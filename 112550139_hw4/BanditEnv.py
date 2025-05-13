import numpy as np

class BanditEnv:
    """
    k-armed (Gaussian) Bandit environment.
    支援：
      • stationary=True  → μ 固定
      • stationary=False → 每步 μ 隨 N(0,0.01²) 隨機漫步
    """
    def __init__(self, k: int, stationary: bool = True,
                 walk_std: float = 0.01, seed: int | None = None):
        assert k > 0, "k must be positive"
        self.k = k       
        self.stationary = stationary
        # Standard deviation of random walk for non-stationary case
        # μ ~ N(0, 0.01²) in this case
        self.walk_std = walk_std
        self.random_state = np.random.RandomState(seed)
        self.actions = []
        self.rewards = []
        self.reset()

    # ---------- public  ----------
    def reset(self) -> None:
        # μ ~ N(0, 1²)
        self.mu = self.random_state.normal(loc=0.0, scale=1.0, size=self.k)
        self.actions.clear()
        self.rewards.clear()

    def step(self, action: int) -> float:
        if not (0 <= action < self.k):
            raise IndexError("action out of range")
        # Non-stationary: μ random walk
        if not self.stationary:
            # μ ← μ + N(0, 0.01²)
            self.mu += self.random_state.normal(
                loc=0.0, scale=self.walk_std, size=self.k)
        reward = self.random_state.normal(loc=self.mu[action], scale=1.0)
        self.actions.append(action)
        self.rewards.append(float(reward))
        return float(reward)

    def export_history(self):
        return list(self.actions), list(self.rewards)

