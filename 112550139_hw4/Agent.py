import numpy as np

class Agent:
    """
    ε-greedy action-value agent。
    參數
    ------
    k : int              # 臂數
    epsilon : float      # 探索率 (ε)
    alpha : float|None   # 常數步長; None → 採樣平均
    seed : int|None
    """
    def __init__(self, k: int, epsilon: float,
                 alpha: float | None = None, seed: int | None = None):
        self.k = k
        self.epsilon = epsilon
        self.alpha = alpha
        self.random_state = np.random.RandomState(seed)
        self.reset()

    # ---------- interaction ----------
    def select_action(self) -> int:
        """依 ε-greedy 規則選擇一臂。"""
        if self.random_state.rand() < self.epsilon:
            return self.random_state.randint(self.k)
        # 若多臂並列最佳，以隨機打破同分
        max_q = np.max(self.Q)
        best_actions = np.flatnonzero(self.Q == max_q)
        return int(self.random_state.choice(best_actions))

    def update_q(self, action: int, reward: float):
        """根據觀測 reward 更新 Q̂。"""
        self.N[action] += 1
        if self.alpha is None:               # ----- sample‑average -----
            step_size = 1.0 / self.N[action]
        else:                                # ----- constant α ---------
            step_size = self.alpha
        self.Q[action] += step_size * (reward - self.Q[action])

    def reset(self):
        """清空估計值與計數。"""
        self.Q = np.zeros(self.k, dtype=np.float64)
        self.N = np.zeros(self.k, dtype=np.int64)
