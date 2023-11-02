import numpy as np
import random
from examples import *
from time_integrators import *

# ... Definitions for Lorenz63 and other functions ...

# Parameters for the Lorenz63 twin experiment
sigma = 10.0     
beta = 8.0/3.0
rho = 28.0     
dt = 0.01
tm = 2.0
nt = int(tm/dt)
t = np.linspace(0, tm, nt+1)
u0True = np.array([1,1,1])
H = np.eye(3)

# Generate uTrue
uTrue = np.zeros([3, nt+1])
uTrue[:, 0] = u0True
for k in range(nt):
    uTrue[:, k+1] = RK4(Lorenz63, uTrue[:, k], dt, sigma, beta, rho)

# Choose a fixed sigma_m for generating observations
true_sigma_m = 0.5
observations = H @ uTrue + np.random.normal(0, true_sigma_m, uTrue.shape)


class LorenzEnvironment:
    def __init__(self, uTrue, observations, steps_per_episode=20, reward_type='var_residuals'):
        self.uTrue = uTrue
        self.observations = observations
        self.current_step = 0
        self.steps_per_episode = steps_per_episode
        self.reward_type = reward_type
    
    def step(self, guessed_sigma_m):
        end_step = min(self.current_step + self.steps_per_episode, self.uTrue.shape[1])
        
        if self.reward_type == 'var_residuals':
            residuals = []
            for step in range(self.current_step, end_step):
                synthetic_obs = H @ self.uTrue[:, step] + np.random.normal(0, guessed_sigma_m, [3,])
                residuals.extend(synthetic_obs - self.observations[:, step])
            reward = -np.var(residuals)
            
        elif self.reward_type == 'direct_noise':
            true_noise = self.observations[:, self.current_step:end_step] - self.uTrue[:, self.current_step:end_step]
            variance_of_true_noise = np.var(true_noise)
            reward = -abs(variance_of_true_noise - guessed_sigma_m**2)
            
        elif self.reward_type == 'likelihood':
            residuals = []
            for step in range(self.current_step, end_step):
                synthetic_obs = H @ self.uTrue[:, step] + np.random.normal(0, guessed_sigma_m, [3,])
                residuals.extend(synthetic_obs - self.observations[:, step])
            likelihood = np.exp(-np.sum(np.array(residuals)**2) / (2 * guessed_sigma_m**2))
            reward = -np.log(likelihood)
            
        elif self.reward_type == 'abs_error':
            true_noise = self.observations[:, self.current_step:end_step] - self.uTrue[:, self.current_step:end_step]
            variance_of_true_noise = np.var(true_noise)
            reward = -abs(np.sqrt(variance_of_true_noise) - guessed_sigma_m)
        
        self.current_step = end_step
        done = self.current_step >= len(self.uTrue[0])
        return reward, done
    
    def reset(self):
        self.current_step = 0


# Q learning parameters
alpha = 0.1
gamma = 0.6
epsilon = 0.4

# Discretize sigma_m space
actions = np.linspace(0.1, 2, 20)

# Initialize Q-values
q_table = np.zeros(len(actions))

# env = LorenzEnvironment(uTrue)
env = LorenzEnvironment(uTrue, observations, reward_type='abs_error')

episode_rewards = []

for episode in range(1000):
    env.reset()  # Reset the environment
    done = False
    total_episode_reward = 0
    action_counts = np.zeros(len(actions))
    while not done:
        if random.uniform(0, 1) < epsilon:
            action_idx = np.random.choice(len(actions))
        else:
            action_idx = np.argmax(q_table)
        reward, done = env.step(actions[action_idx])
        total_episode_reward += reward
        action_counts[action_idx] += 1
        old_value = q_table[action_idx]
        next_max = np.max(q_table)
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[action_idx] = new_value
    
    episode_rewards.append(total_episode_reward)
    if episode % 50 == 0:  # print info every 50 episodes
        print(f"Episode {episode} - Total Reward: {total_episode_reward}")
        print("Action frequencies:", action_counts)
        print("Current best guessed sigma_m:", actions[np.argmax(q_table)])

print("Best guessed sigma_m:", actions[np.argmax(q_table)])
print("True sigma_m:", true_sigma_m)
