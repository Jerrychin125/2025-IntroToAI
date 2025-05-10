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
        self.walk_std = walk_std
        self.random_state = np.random.RandomState(seed)
        self._init_arms()
        self.reset()

    # ---------- public  ----------
    def reset(self) -> None:
        """清空歷史並重新抽 reward distribution 的均值。"""
        self._init_arms()
        self.actions, self.rewards = [], []

    def step(self, action: int) -> float:
        """拉動指定 arm; 回傳 reward。"""
        if not (0 <= action < self.k):
            raise IndexError("action out of range")
        # 非定常環境：先讓所有 μ 作隨機漫步
        if not self.stationary:
            self.mu += self.random_state.normal(
                loc=0.0, scale=self.walk_std, size=self.k)
        reward = self.random_state.normal(loc=self.mu[action], scale=1.0)
        self.actions.append(action)
        self.rewards.append(float(reward))
        return float(reward)

    def export_history(self):
        return list(self.actions), list(self.rewards)

    # ---------- private ----------
    def _init_arms(self):
        self.mu = self.random_state.normal(loc=0.0, scale=1.0, size=self.k)
