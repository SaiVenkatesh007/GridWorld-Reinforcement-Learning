import matplotlib.pyplot as plt
import numpy as np

def train_agent(agent, env, episodes=1000):
    for episode in range(episodes):
        state = env.reset()
        finished = False
        while not finished:
            action = agent.action_choice(state)
            next_state, reward, finished = env.step(env.actions[action])
            agent.q_table_update(state, action, reward, next_state)
            state = next_state

def path_visualization(env, path):
    grid = np.zeros((env.size, env.size))
    for obstacle in env.obstacles:
        grid[obstacle[0], obstacle[1]] = -1
    for state in path:
        grid[state[0], state[1]] = -1
    grid[env.start[0], env.start[1]] = 0.5
    grid[env.goal[0], env.goal[1]] = 0.5
    plt.imshow(grid, cmap='viridis', origin='lower')
    plt.show()

def test_agent(agent, env, episodes=100, visualize=False):
    total_rewards = []
    for _ in range(episodes):
        state = env.reset()
        finished = False
        total_reward = 0
        path = [state]
        while not finished:
            action = agent.action_choice(state)
            next_state, reward, finished = env.step(env.actions[action])
            total_reward += reward
            state = next_state
            path.append(state)
        total_rewards.append(total_reward)
        if visualize:
            path_visualization(env, path)
    return total_rewards