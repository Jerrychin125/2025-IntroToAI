class Agent:
    def __init__(self, name):
        self.name = name
        self.state = None
        self.action = None

    def set_state(self, state):
        self.state = state

    def get_state(self):
        return self.state

    def set_action(self, action):
        self.action = action

    def get_action(self):
        return self.action

    def __str__(self):
        return f"Agent {self.name} with state {self.state} and action {self.action}"
    
    def update_q(self, reward):
        # Placeholder for Q-value update logic
        pass
    
    def select_action(self):
        # Placeholder for action selection logic
        return self.action