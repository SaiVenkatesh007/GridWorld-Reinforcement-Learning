import numpy as np

class LearningAgent:
    def __init__(self, num_actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}

    def action_choice(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.randint(0, self.num_actions)
        else:
            return np.argmax(self.q_table.get(state, np.zeros(self.num_actions)))

    def q_table_update(self, state, action, reward, next_state):
        q_current = self.q_table.get(state, np.zeros(self.num_actions))
        q_next = self.q_table.get(next_state, np.zeros(self.num_actions))
        target = reward + self.gamma*np.max(q_next)
        error = target - q_current[action]
        q_current[action] += self.alpha*error
        self.q_table[state] = q_current
