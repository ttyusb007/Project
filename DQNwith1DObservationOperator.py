import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from collections import deque
import matplotlib.pyplot as plt

def Lorenz63(state, *args):
    sigma = args[0]
    beta = args[1]
    rho = args[2]
    x, y, z = state
    f = np.zeros(3)
    f[0] = sigma * (y - x)
    f[1] = x * (rho - z) - y
    f[2] = x * y - beta * z
    return f

def RK4(func, y, h, *args):
    k1 = h * func(y, *args)
    k2 = h * func(y + 0.5 * k1, *args)
    k3 = h * func(y + 0.5 * k2, *args)
    k4 = h * func(y + k3, *args)
    return y + (k1 + 2*k2 + 2*k3 + k4) / 6

def observe(state, H):
    """Observation function that projects the 3D state to 1D using observation operator H"""
    return np.dot(H, state)

dt = 0.01
tm = 2.0
nt = int(tm / dt)
t = np.linspace(0, tm, nt + 1)
u0True = np.array([1, 1, 1])
H_true = np.array([0.5, 0.3, 0.2]) # True observation operator
sigma = 10.0
beta = 8.0 / 3.0
rho = 28.0

uTrue = np.zeros([3, nt + 1])
uTrue[:, 0] = u0True
for k in range(nt):
    uTrue[:, k + 1] = RK4(Lorenz63, uTrue[:, k], dt, sigma, beta, rho)

noise_std = 0.5
observations = np.array([observe(uTrue[:, k], H_true) + np.random.normal(0, noise_std) for k in range(nt + 1)])

action_space = [(0.1, 0, 0), (-0.1, 0, 0), (0, 0.1, 0), (0, -0.1, 0), (0, 0, 0.1), (0, 0, -0.1), (0, 0, 0)]

class LorenzEnvironment:
    def __init__(self, uTrue, observations, penalty_factor=0.001):
        self.uTrue = uTrue
        self.observations = observations
        self.penalty_factor = penalty_factor
        self.state_size = 3 + 2*(nt + 1)  # Parameters H
        # Parameters H + Observations + Predicted Observations
        self.action_space = action_space
        self.H_parameters = np.array([1.2, 1.2, 1.2]) # Initialize H parameters
        self.max_steps = 135
        self.current_step = 0

    def step(self, action_idx):
        action = self.action_space[action_idx]
        guessed_H_params = self.H_parameters + np.array(action)
        
        predicted_observations = np.array([observe(self.uTrue[:, step], guessed_H_params) for step in range(len(self.uTrue[0]))])

        abs_loss = np.abs(predicted_observations - self.observations)
        mse = np.mean(abs_loss ** 2)

        next_state = np.concatenate((guessed_H_params, self.observations, predicted_observations))

        parameter_change_penalty = np.sum(np.abs(self.H_parameters - guessed_H_params))
        reward = -mse # + self.penalty_factor * parameter_change_penalty
        self.H_parameters = guessed_H_params
        
        self.current_step += 1
        
        if self.current_step >= self.max_steps:
            done = True
        else:
            done = False
        return next_state, reward, done
    
    def reset(self):
        self.current_step = 0
        return np.zeros(self.state_size)

env = LorenzEnvironment(uTrue, observations, penalty_factor=0.001)
state_size = env.state_size
action_size = len(env.action_space)

class DQLAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.998
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.losses = []

    def _build_model(self):
        model = keras.Sequential()
        model.add(keras.layers.Dense(64, activation='relu', input_shape=(self.state_size,)))
        model.add(keras.layers.Dense(64, activation='relu'))
        model.add(keras.layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(np.array([state]))
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        states = np.array([x[0] for x in minibatch])
        next_states = np.array([x[3] for x in minibatch])

        # print("states shape:", states.shape) # DEBUG
        # print("next_states shape:", next_states.shape) # DEBUG


        q_values = self.model.predict(states)
        q_values_next = self.model.predict(next_states)

        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            if not done:
                q_values[i][action] = reward + self.gamma * np.amax(q_values_next[i])
            else:
                q_values[i][action] = reward

        # Train the model using the states and updated Q values
        # and store the loss value from the training
        history = self.model.fit(states, q_values, epochs=1, verbose=0)
        self.losses.append(history.history['loss'][0])


        self.model.fit(states, q_values, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

agent = DQLAgent(state_size, action_size)
batch_size = 32

episode_rewards = []
best_guessed_H = []
mse_values = []

for episode in range(100):
    state = env.reset()
    done = False
    total_episode_reward = 0
    while not done:
        action_idx = agent.act(state)
        next_state, reward, done = env.step(action_idx)
        total_episode_reward += reward
        agent.remember(state, action_idx, reward, next_state, done)
        state = next_state

        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

    episode_rewards.append(total_episode_reward)
    best_guessed_H.append(env.H_parameters)
    mse_current = np.mean((env.H_parameters - H_true) ** 2)
    mse_values.append(mse_current)


    if episode % 10 == 0:
        print(f"Episode {episode} - Total Reward: {total_episode_reward}")

best_action_idx = np.argmax(agent.model.predict(np.array([env.reset()])))
best_H_parameters = env.H_parameters
print("Best guessed observation operator parameters:", best_H_parameters)

# Running average of rewards over episodes
window_size = 10
running_avg = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
plt.plot(running_avg)
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.title(f'Running Average of Rewards (Window Size: {window_size})')
plt.grid(True)
plt.show()

# Plotting the loss over training steps
plt.figure()
plt.plot(agent.losses)
plt.xlabel('Training Step')
plt.ylabel('Loss')
plt.title('Loss over Training Steps')
plt.grid(True)
plt.show()

# Mean Squared Error of Guessed H Parameters
plt.figure()
plt.plot(mse_values)
plt.xlabel('Episode')
plt.ylabel('MSE')
plt.title('Mean Squared Error of Guessed H Parameters')
plt.grid(True)
plt.show()
