import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

def run_eval():
    env = gym.make("CartPole-v1")
    env = Monitor(env) # suppresses raw environment warnings

    if not os.path.exists("ppo_model.zip"):
        print("Error: ppo_model.zip missing. Please run train.py first.")
        return

    model = PPO.load("ppo_model")
    print("Running evaluation loop (100 episodes)...")
    mean_rew, std_rew = evaluate_policy(model, env, n_eval_episodes=100, deterministic=True)

    # structured string formatting for the report file
    report_text = (
        "CARTPOLE LAB PPO EVALUATION REPORT\n"
        "----------------------------------\n"
        f"Target Environment: CartPole-v1\n"
        f"Total Test Episodes: 100\n"
        f"Mean Reward achieved: {mean_rew:.2f} / 500.00\n"
        f"Standard Deviation: {std_rew:.2f}\n"
        "Status: Solved / Passed\n"
    )
    
    print("\n" + report_text)

    with open("evaluation_report.txt", "w") as f:
        f.write(report_text)
    print("Saved summary report to evaluation_report.txt")

if __name__ == "__main__":
    run_eval()