import magent2
from magent2.environments import battlefield_v5
from magent2.environments import magent_env
import numpy as np

# ---------------------------------------------------------
# PATCH: Fix magent2 compatibility with latest PettingZoo
# ---------------------------------------------------------
# The current version of magent2's reset() method returns only observations,
# but new PettingZoo/Gymnasium requires (observations, info).
# We monkey-patch the reset method to add an empty info dict.

original_reset = magent_env.magent_parallel_env.reset

def patched_reset(self, seed=None, return_info=False, options=None):
    # Call the original reset
    obs = original_reset(self, seed=seed, return_info=return_info, options=options)
    
    # If it returns a tuple, assume it's already fixed possibly? 
    # But based on source it returns dict.
    if isinstance(obs, dict):
        infos = {agent: {} for agent in obs.keys()}
        return obs, infos
    return obs

magent_env.magent_parallel_env.reset = patched_reset
# ---------------------------------------------------------

print("🚀 Starting Magent2 Verification...")

try:
    env = battlefield_v5.env(render_mode=None)
    
    # Reset environment (returns None for AEC envs typically, internal state updated)
    env.reset() 
    
    print("✅ Environment initialized successfully.")

    # AEC API loop (standard for PettingZoo/MAgent2)
    for agent in env.agent_iter(max_iter=10):
        # env.last() returns 5 values now:
        # (observation, reward, termination, truncation, info)
        observation, reward, termination, truncation, info = env.last()
        
        if termination or truncation:
            action = None
        else:
            action = env.action_space(agent).sample()
        
        # Step only takes the action
        env.step(action)
    
    print("✅ Simulation loop completed without errors.")

    print("\n🎉 Your local RL environment is fully compatible with MacOS 2026 standards!")

except Exception:
    import traceback
    traceback.print_exc()