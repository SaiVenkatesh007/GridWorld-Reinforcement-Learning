class GridWorld:
    def __init__(self, size=10, start=(10,10), goal=(9,9), obstacles=[]):
        self.size = size
        self.start = start
        self.goal = goal
        self.obstacles = obstacles
        # actions -> right, down, left, up
        self.actions = [(0,1), (1,0), (0,-1), (-1,0)]
        self.num_actions = len(self.actions)
        self.state = start

    def reset(self):
        self.state = self.start

    def is_valid(self, state):
        x, y = state
        valid = 0 <= x < self.size and 0 <= y < self.size and state not in self.obstacles
        return valid

    def step(self, action):
        next_state = (self.state[0] + action[0], self.state[1] + action[1])
        if self.is_valid(next_state):
            self.state = next_state
        reward = -1 if self.state != self.goal else 0
        done = self.state == self.goal
        return self.state, reward, done

    def get_state(self):
        return self.state

    def get_size(self):
        return self.size
