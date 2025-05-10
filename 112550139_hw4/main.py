import numpy as np
import matplotlib.pyplot as plt
from BanditEnv import BanditEnv
from Agent import Agent

N_RUNS      = 2000
STEPS_STAT  = 1000
STEPS_NONST = 10_000
K           = 10

def run_bandit(env_cfg, agent_cfg, steps):
    rewards   = np.zeros((N_RUNS, steps))
    optimal   = np.zeros_like(rewards, dtype=bool)

    for run in range(N_RUNS):
        env   = BanditEnv(**env_cfg)
        agent = Agent(**agent_cfg)
        q_opt = np.argmax(env.mu)  # true optimal (for stationary env)  
        for t in range(steps):
            action = agent.select_action()
            r      = env.step(action)
            agent.update_q(action, r)
            rewards[run, t] = r
            # 判斷是否選到當下最優臂
            if env_cfg["stationary"]:
                optimal[run, t] = (action == q_opt)
            else:                                   # 非定常：即時比較
                optimal[run, t] = (action == np.argmax(env.mu))

    return rewards.mean(0), optimal.mean(0)  # shape:(steps,)

def plot_curves(x, curves, labels, title, ylabel, filename):
    plt.figure(figsize=(7,4))
    for y, lbl in zip(curves, labels):
        plt.plot(x, y, label=lbl)
    plt.xlabel("Steps")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def experiment_stationary():
    epsilons = [0.0, 0.01, 0.1]
    avg_r, pct_opt = [], []
    for eps in epsilons:
        r, opt = run_bandit(
            env_cfg = dict(k=K, stationary=True, seed=None),
            agent_cfg = dict(k=K, epsilon=eps, alpha=None, seed=None),
            steps = STEPS_STAT)
        avg_r.append(r)
        pct_opt.append(opt)
    x = np.arange(STEPS_STAT)
    plot_curves(x, avg_r , [f"ε={e}" for e in epsilons],
                "Average Reward (stationary)", "Avg reward",
                "part3_reward.png")
    plot_curves(x, pct_opt, [f"ε={e}" for e in epsilons],
                "% Optimal Action (stationary)", "Optimal action %",
                "part3_opt.png")

def experiment_nonstationary_sampleavg():
    r, opt = run_bandit(
        env_cfg  = dict(k=K, stationary=False, seed=None),
        agent_cfg= dict(k=K, epsilon=.1, alpha=None, seed=None),
        steps    = STEPS_NONST)
    x = np.arange(STEPS_NONST)
    plot_curves(x, [r], ["ε=0.1, sample‑avg"],
                "Avg Reward (non‑stationary, sample‑avg)",
                "Avg reward", "part5_reward.png")
    plot_curves(x, [opt], ["ε=0.1, sample‑avg"],
                "% Optimal (non‑stationary, sample‑avg)",
                "Optimal action %", "part5_opt.png")

def experiment_nonstationary_constalpha():
    alphas = [0.1, 0.01]
    avg_r, pct_opt = [], []
    for a in alphas:
        r, opt = run_bandit(
            env_cfg  = dict(k=K, stationary=False, seed=None),
            agent_cfg= dict(k=K, epsilon=.1, alpha=a, seed=None),
            steps    = STEPS_NONST)
        avg_r.append(r)
        pct_opt.append(opt)
    x = np.arange(STEPS_NONST)
    plot_curves(x, avg_r , [f"α={a}" for a in alphas],
                "Avg Reward (non‑stationary, const α)",
                "Avg reward", "part7_reward.png")
    plot_curves(x, pct_opt, [f"α={a}" for a in alphas],
                "% Optimal (non‑stationary, const α)",
                "Optimal action %", "part7_opt.png")

if __name__ == "__main__":
    experiment_stationary()
    experiment_nonstationary_sampleavg()
    experiment_nonstationary_constalpha()
    print("All figures saved: part3_*.png, part5_*.png, part7_*.png")
