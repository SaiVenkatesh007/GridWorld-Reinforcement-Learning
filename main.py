from learning_agent import LearningAgent
from grid_world import GridWorld
from utils import train_agent, test_agent, path_visualization
import numpy as np

env = GridWorld(size=10, start=(0,0), goal=(9,9), obstacles=[(3, 3), (3, 4), (3, 5)])
agent = LearningAgent(num_actions=4)

train_agent(agent=agent, env=env)
rewards, path = test_agent(agent=agent, env=env, visualize=True)
print(f"Average Rewards {np.mean(rewards)}")
print(f"Path is \n {path}")