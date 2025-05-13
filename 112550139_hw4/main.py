import numpy as np
import matplotlib.pyplot as plt
from BanditEnv import BanditEnv
from Agent import Agent

N_RUNS      = 2000
STEPS_STAT  = 1000
STEPS_NONST = 10000
K           = 10
EPSILON     = [0.0, 0.01, 0.1]

def run_bandit(env_cfg, agent_cfg, steps):
    rewards   = np.zeros((N_RUNS, steps))
    optimal   = np.zeros_like(rewards, dtype=bool)

    for run in range(N_RUNS):
        env   = BanditEnv(**env_cfg)
        agent = Agent(**agent_cfg)
        q_opt = np.argmax(env.mu)
        for t in range(steps):
            action = agent.select_action()
            r      = env.step(action)
            agent.update_q(action, r)
            rewards[run, t] = r
            if env_cfg["stationary"]:
                optimal[run, t] = (action == q_opt)
            else:
                optimal[run, t] = (action == np.argmax(env.mu))

    return rewards.mean(0), optimal.mean(0)

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
    avg_r, pct_opt = [], []
    for eps in EPSILON:
        r, opt = run_bandit(
            env_cfg = dict(k=K, stationary=True),
            agent_cfg = dict(k=K, epsilon=eps, alpha=None),
            steps = STEPS_STAT)
        avg_r.append(r)
        pct_opt.append(opt)
    x = np.arange(STEPS_STAT)
    plot_curves(x, avg_r , [f"ε={e}" for e in EPSILON],
                "Average Reward (stationary)", "Avg reward",
                "part3_reward.png")
    plot_curves(x, pct_opt, [f"ε={e}" for e in EPSILON],
                "% Optimal Action (stationary)", "Optimal action %",
                "part3_opt.png")

def experiment_nonstationary_sampleavg():
    avg_r, pct_opt = [], []
    for eps in EPSILON:
        r, opt = run_bandit(
            env_cfg  = dict(k=K, stationary=False),
            agent_cfg= dict(k=K, epsilon=eps, alpha=None),
            steps    = STEPS_NONST)
        avg_r.append(r)
        pct_opt.append(opt)
    x = np.arange(STEPS_NONST)
    plot_curves(x, avg_r, [f"ε={e}, sample-avg" for e in EPSILON],
                "Avg Reward (non-stationary, sample-avg)",
                "Avg reward", "part5_reward.png")
    plot_curves(x, pct_opt, [f"ε={e}, sample-avg" for e in EPSILON],
                "% Optimal (non-stationary, sample-avg)",
                "Optimal action %", "part5_opt.png")

def experiment_nonstationary_constalpha():
    avg_r, pct_opt = [], []
    for eps in EPSILON:
        r, opt = run_bandit(
            env_cfg  = dict(k=K, stationary=False),
            agent_cfg= dict(k=K, epsilon=eps, alpha=0.1),
            steps    = STEPS_NONST)
        avg_r.append(r)
        pct_opt.append(opt)
    x = np.arange(STEPS_NONST)
    plot_curves(x, avg_r , [f"ε={e}" for e in EPSILON],
                "Avg Reward (non‑stationary, const α)",
                "Avg reward", "part7_reward.png")
    plot_curves(x, pct_opt, [f"ε={e}" for e in EPSILON],
                "% Optimal (non‑stationary, const α)",
                "Optimal action %", "part7_opt.png")

def experiment_nonstationary_multialpha():
    alphas = [0.1, 0.01]
    avg_r, pct_opt = [], []
    for a in alphas:
        r, opt = run_bandit(
            env_cfg  = dict(k=K, stationary=False),
            agent_cfg= dict(k=K, epsilon=0, alpha=a),
            steps    = STEPS_NONST)
        avg_r.append(r)
        pct_opt.append(opt)
    x = np.arange(STEPS_NONST)
    plot_curves(x, avg_r , [f"α={a}" for a in alphas],
                "Avg Reward (non‑stationary, const α)",
                "Avg reward", "disc_reward.png")
    plot_curves(x, pct_opt, [f"α={a}" for a in alphas],
                "% Optimal (non‑stationary, const α)",
                "Optimal action %", "disc_opt.png")

if __name__ == "__main__":
    experiment_stationary()
    experiment_nonstationary_sampleavg()
    experiment_nonstationary_constalpha()
    experiment_nonstationary_multialpha()
    print("All figures saved: part3_*.png, part5_*.png, part7_*.png, disc_*.png")
