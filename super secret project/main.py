import os
import numpy as np
from data_manager import get_prices, plot_prices
from model_manager import check_and_load_model, save_model
from trading_environment import TradingEnv
from q_learning_agent import QLearningAgent
from a2c_agent import A2CAgent
path = "C:\\Users\\Computing\\Desktop\\super secret project" # Root folder containing the model
model_filename = "model.keras"
model_path = os.path.join(path, model_filename)

def main():
    # Fetch Stock Data
    symbol = 'AAPL'
    start_date = '2024-01-01'
    end_date = '2024-02-28'
    stock_prices = get_prices(symbol, start_date, end_date)

    # Initialize Environment and Agent
    env = TradingEnv(data=stock_prices)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    model = check_and_load_model(model_path, input_shape=state_size, action_size=action_size)
    #agent = QLearningAgent(model=model, action_size=action_size)

    # QLearningAgent Training Loop
    num_episodes = 10  # Number of episodes for training
    # for e in range(num_episodes):
    #     state = env.reset()
    #     state = np.reshape(state, [1, env.lookback_window_size])  # Reshape state to match the input shape of the model
    #     for time in range(60):
    #         action = agent.act(state)
    #         next_state, reward, done, _ = env.step(action)
    #         next_state = np.reshape(next_state, [1, env.lookback_window_size])
    #         agent.learn(state, action, reward, next_state, done)
    #         state = next_state
    #         if done:
    #             break
    #     print(f"Epoch {e+1} Completed!")

    agent = A2CAgent(state_size = env.observation_space.shape[0], action_size = env.action_space.n)

    for episode in range(num_episodes):
        env.render(mode='file', save=True, epoch = episode)
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.act(state)
            next_state, reward, done, _  = env.step(action)
            agent.learn(state,action,reward,next_state, done)
            state = next_state
            total_reward += reward


        print(f"Episode: {episode+1}, Total Reward: {total_reward}")


    save_model(model, path)
    plot_prices(stock_prices)
    print("Training completed!")

if __name__ == "__main__":
    main()