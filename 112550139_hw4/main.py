from BanditEnv import BanditEnv
from Agent import Agent
import random

# Exmple behavior of the game environment

# BanditEnv 
# Example behavior of the BanditEnv class
env = BanditEnv()
env.reset()

actions = []
rewards = []
for _ in range(1000):
    action = random.randint(0, 9) 
    reward = env.step(action)
    actions.append(action)
    rewards.append(reward)

action_history, reward_history = env.get_history()

for i in range(1000):
    assert action_history[i] == actions[i]
    assert reward_history[i] == rewards[i]

# Agent
# Example behavior of the Agent class
k = 10
epsilon = 0.1
agent = Agent(k, epsilon)
action = agent.select_action()
reward = 0
agent.update_q(action, reward)

"""

"""