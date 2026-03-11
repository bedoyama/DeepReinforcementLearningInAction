import gymnasium as gym

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
            observation, info = env.reset()

except KeyboardInterrupt:
    print("Simulation stopped by user")
finally:
    env.close()
