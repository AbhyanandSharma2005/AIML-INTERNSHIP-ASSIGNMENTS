import gymnasium as gym
from stable_baselines3 import PPO

def render_agent():
    # render with human mode to physically view progress window
    env = gym.make("CartPole-v1", render_mode="human")
    model = PPO.load("ppo_model")

    obs, info = env.reset()
    done = False
    score = 0

    print("Opening visual simulation window...")
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        score += reward
        done = terminated or truncated

    print(f"Simulation loop ended. Final Score: {score}")
    env.close()

if __name__ == "__main__":
    render_agent()