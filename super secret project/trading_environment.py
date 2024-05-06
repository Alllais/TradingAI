import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import os

class TradingEnv(gym.Env):
    """A simple trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, data, debug = False, initial_balance=500, lookback_window_size=200, asset_holding = 0):
        super(TradingEnv, self).__init__()

        self.illegal_buy_attempts = 0
        self.illegal_sell_attempts = 0
        self.illegal_hold_attempts = 0

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
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(lookback_window_size, 10), dtype=np.float32)

        # For rendering
        self.rendering_data = {'prices' : [], 'actions' : []}

    def step(self, action):
        self.n_step += 1
        done = self.n_step >= len(self.data) - 1
        current_price = self.data[self.n_step]
        reward = 0

        if action == 0: # Hold
          self.illegal_hold_attempts += 15
          reward -= self.illegal_hold_attempts
          pass

        elif action == 1:  # Buy
            if self.balance >= current_price:
                self.balance -= current_price
                self.asset_holding += 1
                self.illegal_buy_attempts = 0
                self.illegal_hold_attempts = 0
            else:
                reward -= 2 * self.illegal_buy_attempts  # Penalty for trying to buy without enough balance
                self.illegal_buy_attempts += 10

        elif action == 2:  # Sell
            if self.asset_holding > 0:
                self.balance += current_price
                self.asset_holding -= 1
                self.illegal_sell_attempts = 0
                self.illegal_hold_attempts = 0
            else:
                reward -= 2 * self.illegal_sell_attempts  # Penalty for trying to sell without holding any assets
                self.illegal_sell_attempts += 10

        net_worth = self.balance + (self.asset_holding * current_price)

        #if (net_worth >= self.initial_balance):
          #reward += (net_worth - self.initial_balance)
        #else:
        #  reward -= 50
          
        # Update reward based on the change in net worth
        # Dynamic Reward system
        #reward += net_worth - self.initial_balance
        #if(net_worth == self.initial_balance):
        #  reward -= 1000

        # change the worth based on each step
        
        reward += (net_worth - self.last_net_worth) * 10

        self.last_net_worth = net_worth
        
        if (net_worth == 495.83): reward=-1000
        
        #if (net_worth > )


        #if (reward < 0): #half the downside from a loss
        #  reward /= 2
        #elif (reward > 0): # double the reward from a gain
        #  reward *= 2

        # take market context into account when rewarding the agent
        #if action == 1 and self.balance > current_price * 1.1:  # Buying with a buffer
        #  reward += 10  # Encourage buying with a safety margin
        #elif action == 2 and self.asset_holding > 1:
        #  reward += 10  # Encourage selling when holding multiple assets

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
            save_path = "C:\\Users\\Computing\\Desktop\\super secret project\\Images"
            plot_filename = f'trading_plot_epoch_{epoch}.png'
            full_plot_path = os.path.join(save_path, plot_filename)
            plt.savefig(full_plot_path)
            print(f"Plot for epoch {epoch} saved.")

        #plt.show()
        if close:
           plt.close()