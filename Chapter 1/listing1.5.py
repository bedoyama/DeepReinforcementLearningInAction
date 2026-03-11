import gymnasium as gym
import argparse

parser = argparse.ArgumentParser(description='Run CarRacing with random actions.')
parser.add_argument('--no-reset', action='store_true', help='Do not reset the environment when the episode ends')
args = parser.parse_args()

# Create the environment with render_mode="human" to see the window
env = gym.make('CarRacing-v2', render_mode="human")

# Reset returns observation and info
observation, info = env.reset()

# Loop to keep the window open and simulation running
try:
    while True:
        # Sample a random action (no control policy)
        action = env.action_space.sample()
        
        # Step returns 5 values in Gymnasium
        observation, reward, terminated, truncated, info = env.step(action)
        
        # Reset if the episode ends
        if terminated or truncated:
            if not args.no_reset:
                observation, info = env.reset()
            else:
                pass

except KeyboardInterrupt:
    print("Simulation stopped by user")
finally:
    env.close()
