import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf
import numpy as np
import random
import gym
from gym import spaces
import pandas as pd
import os
import pickle

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

path = "C:\\Users\\Computing\\Desktop\\TradingAI-main" # Root folder containing the model
pathImages = "C:\\Users\\Computing\\Desktop\\TradingAI-main\\images" # File path for the graphs output

# Check for existing model
model_filename = "model.keras"
model_path = os.path.join(path, model_filename)

matplotlib.use('Agg')

def check_and_load_model(model_path, input_shape, action_size):
   if os.path.exists(model_path):
      print("Loading existing model...")
      with open(model_path, 'rb') as file:
        return load_model(model_path)
   else:
      print("No exisitng model found. Creating a new one...")
      return create_q_model(input_shape, action_size)
    
def create_q_model(input_shape, action_size):
    model = Sequential()
    model.add(Dense(24, input_shape=(input_shape,), activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(action_size, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.01))
    return model


# Data
def get_prices (share_symbol, start_date, end_date, cache_filename = "stock_prices.npy"):
  try:
    stock_prices = np.load(cache_filename)
  except IOError:
    stock_data = yf.download(share_symbol, start = start_date, end = end_date)
    stock_prices = stock_data['Open'].values
    np.save(cache_filename, stock_prices)

  return stock_prices.astype(float)

def plot_prices(prices):
  plt.figure(figsize=(8, 4))
  plt.title('Opening Stock Prices')
  plt.xlabel('Day')
  plt.ylabel('Price ($)')
  plt.plot(prices)
  plt.savefig('prices.png')
  #plt.show()

# Environment

class TradingEnv(gym.Env):
    """A simple trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, data, debug = False, initial_balance=10000, lookback_window_size=200, asset_holding = 0):
        super(TradingEnv, self).__init__()

        # Debug mode
        self.debug = debug

        # Historical market data
        self.data = data
        self.lookback_window_size = lookback_window_size

        # Initialize state
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.n_step = 0
        self.asset_holding = 0
        self.last_net_worth = initial_balance

        self.last_epoch_portfolio_value = 0

        # Define action and observation space
        self.action_space = spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(lookback_window_size, 5), dtype=np.float32)

        # For rendering
        self.rendering_data = {'prices' : [], 'actions' : []}

    def step(self, action):
        self.n_step += 1
        done = self.n_step >= len(self.data) - 1
        current_price = self.data[self.n_step]
        reward = 0

        if action == 0: # Hold
          reward -= 5
          pass

        elif action == 1:  # Buy
            if self.balance >= current_price:
                self.balance -= current_price
                self.asset_holding += 1
            else:
                reward -= 20  # Penalty for trying to buy without enough balance

        elif action == 2:  # Sell
            if self.asset_holding > 0:
                self.balance += current_price
                self.asset_holding -= 1
            else:
                reward -= 20  # Penalty for trying to sell without holding any assets

        net_worth = self.balance + (self.asset_holding * current_price)

        #if (net_worth >= self.initial_balance):
        #  reward += 50
        #else:
        #  reward -= 50

        # Update reward based on the change in net worth
        # Dynamic Reward system
        #reward += net_worth - self.initial_balance
        #if(net_worth == self.initial_balance):
        #  reward -= 1000

        # change the worth based on each step
        
        reward += (net_worth - self.last_net_worth)

        self.last_net_worth = net_worth


        

        # take market context into account when rewarding the agent
        if action == 1 and self.balance > current_price * 1.1:  # Buying with a buffer
          reward += 10  # Encourage buying with a safety margin
        elif action == 2 and self.asset_holding > 1:
          reward += 10  # Encourage selling when holding multiple assets

        #Market awarness rewarding strategy
        #if np.mean(self.data[max(0, self.n_step - 5):self.n_step + 1]) < current_price:
        # Price is rising
        #if action == 1:
        #  reward += 15  # Reward buying in a rising market


        # Volatility adjustment
        #volatility = np.std(self.data[max(0, self.n_step - 10):self.n_step + 1])
        #reward += (net_worth - last_net_worth) / volatility



        # Get the next state
        next_state = self._next_observation()
        if self.debug:
            print(f"Step: {self.n_step}, Balance: {self.balance}, Assets: {self.asset_holding}, Net Worth: {net_worth}, Reward: {reward}")

        self.rendering_data['prices'].append(current_price)
        self.rendering_data['actions'].append(action)

        #print("Reward: ", reward)

        return next_state, reward, done, {}

    def reset(self):
        self.balance = self.initial_balance
        self.asset_holding = 0
        self.n_step = 0
        return self._next_observation()

    def _next_observation(self):
        # Initialize an observation array filled with zeros
        observation = np.zeros(self.lookback_window_size)

        # Calculate end index first
        end_index = self.n_step + 1

        # Calculate start index based on end index and lookback window size
        start_index = max(end_index - self.lookback_window_size, 0)

        # Slice the data
        real_observation = self.data[start_index:end_index]

        # Replace the relevant part of the observation with actual data
        observation[-len(real_observation):] = real_observation

        return observation

            # Random Policy
            #return (self.data[start_index:end_index])

    def calculate_portfolio_value(self):
      current_price = self.data[self.n_step] if self.n_step < len(self.data) else self.data[-1]
      return self.balance + (self.asset_holding * current_price)

    def render(self, mode='human', close=False, save=False, epoch=None):
        # Create a new figure for each plot
        plt.figure(figsize=(12, 6))

        prices = self.rendering_data['prices']
        actions = self.rendering_data['actions']

        # Get indices for hold, buy, and sell actions
        hold_idx = [i for i, a in enumerate(actions) if a == 0]
        buy_idx = [i for i, a in enumerate(actions) if a == 1]
        sell_idx = [i for i, a in enumerate(actions) if a == 2]

        plt.plot(prices, label='Open Price', color='blue')
        plt.scatter(hold_idx, [prices[i] for i in hold_idx], label='Hold', marker='o', color='grey')
        plt.scatter(buy_idx, [prices[i] for i in buy_idx], label='Buy', marker='^', color='green')
        plt.scatter(sell_idx, [prices[i] for i in sell_idx], label='Sell', marker='v', color='red')
        plt.title('Trading Actions Over Time')
        plt.xlabel('Step')
        plt.ylabel('Price')
        plt.legend()

        final_portfolio_value = self.calculate_portfolio_value()
        # reward += final_portfolio_value - initial_balance
        plt.figtext(0.3, 0.01, f'Cash balance: ${self.balance:.2f},   Shares: {self.asset_holding:.2f}, Total: ${final_portfolio_value:.2f}', ha = 'center', fontsize = 12, color = 'red')

        # Save the plot with a unique name for each epoch
        if save and epoch is not None:
            save_path = pathImages
            plot_filename = f'trading_plot_epoch_{epoch}.png'
            full_plot_path = os.path.join(save_path, plot_filename)
            plt.savefig(full_plot_path)
            print(f"Plot for epoch {epoch} saved.")

        #plt.show()
        if close:
           plt.close()


def save_model (model, filename = "model.keras"):
   save_path = path
   full_path = os.path.join(save_path, filename)
   with open(full_path, "wb") as file:
      pickle.dump (model, file)

class QLearningAgent:

  # Stop logging
  #tf.keras.utils.disable_interactive_logging()

  def __init__(self, model, action_size):
    self.model = model
    self.state_size = state_size
    self.action_size = action_size
    self.memory = []
    self.gamma = 0.95
    self.epsilon = 1.0
    self.epsilon_min = 0.01
    self.epsilon_decay=0.995

  def act(self, state):
    if np.random.rand() <= self.epsilon:
      return random.randrange(self.action_size)
    act_values = self.model.predict(state)
    return np.argmax(act_values[0])

  def learn(self, state, action, reward, next_state, done):
    target = reward
    if not done:
      target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
    target_f = self.model.predict(state)
    target_f[0][action] = target
    self.model.fit(state, target_f, epochs = 1, verbose = 0)
    if self.epsilon > self.epsilon_min:
      self.epsilon *= self.epsilon_decay

# Step 1: Fetch Stock Data
symbol = 'AAPL'
start_date = '2021-10-01'
end_date = '2022-10-01'
stock_prices = get_prices(symbol, start_date, end_date)

# Step 2: Initialize Environment and Agent
env = TradingEnv(data=stock_prices, debug=False)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
model = check_and_load_model(model_path, input_shape=state_size, action_size=action_size)
agent = QLearningAgent(model = model, action_size=action_size)
# Step 3: Training Loop
num_episodes = 100  # Number of episodes for training

for e in range(num_episodes):
    #env.render()
    env.render(epoch = e, save=True, close=True)
    env.rendering_data = {'prices': [], 'actions': []}
    state = env.reset()
    state = np.reshape(state, [1, env.lookback_window_size])  # Reshape state to match the input shape of the model

    for time in range(250):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, env.lookback_window_size])  # Reshape the next state
        agent.learn(state, action, reward, next_state, done)
        state = next_state
        #env.render(close=True, save=True)


        if done:
            #env.render(save = True, epoch=e, close=True)
            #env.rendering_data = {'prices':[], 'actions':[]}
            break
    print(f"Epoch {e+1} Completed!")

#for e in range(num_episodes):
#   env.render(save = True, epoch = e, close=True)
#   env.rendering_data = {'prices': [], 'actions': []}


save_model(model, "model.keras")
print("Model saved at: ", model_path)
plot_prices(env.data)
print("Training completed!")
