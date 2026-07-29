import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy

def main():
    log_dir = "./logs/"
    os.makedirs(log_dir, exist_ok=True)

    # set up env with monitor to save episode data for the curve
    env = gym.make("CartPole-v1")
    env = Monitor(env, log_dir)

    # initialize policy network and ppo
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4, n_steps=2048)
    
    print("--- Training Started ---")
    model.learn(total_timesteps=50000)
    print("--- Training Complete ---")

    # save weights matching required directory structure
    model.save("ppo_model")
    print("Saved weights to ppo_model.zip")

    # parse logs to plot training progress
    timesteps, rewards = ts2xy(load_results(log_dir), "timesteps")
    
    # basic rolling window to smooth out the curve chart
    window_size = 10
    if len(rewards) >= window_size:
        smoothed_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        smoothed_timesteps = timesteps[window_size-1:]
    else:
        smoothed_rewards = rewards
        smoothed_timesteps = timesteps

    plt.figure(figsize=(8, 5))
    plt.plot(smoothed_timesteps, smoothed_rewards, label="Reward (SMA 10)", color='b')
    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    plt.title("CartPole PPO Training Progress")
    plt.grid(True)
    plt.legend()
    plt.savefig("learning_curve.png")
    print("Saved plot to learning_curve.png")

if __name__ == "__main__":
    main()