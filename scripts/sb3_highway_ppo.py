import gymnasium as gym
import torch as th
from stable_baselines3 import PPO
from torch.distributions import Categorical
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv
import highway_env

from sb3_highway_ppo_transformer import write_module_hierarchy_to_file
# ==================================
#        Main script
# ==================================

if __name__ == "__main__":
    train = False
    if train:
        n_cpu = 50
        batch_size = 64
        env = make_vec_env("highway-v0", n_envs=n_cpu, vec_env_cls=SubprocVecEnv)
        model = PPO("MlpPolicy",
                    env,
                    policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
                    n_steps=batch_size * 12 // n_cpu,
                    batch_size=batch_size,
                    n_epochs=10,
                    learning_rate=5e-4,
                    gamma=0.8,
                    verbose=2,
                    tensorboard_log="highway_ppo/")
        # Train the agent
        model.learn(total_timesteps=int(2e5))
        # Save the agent
        model.save("highway_ppo/model")
    else:
        device = torch.device("cpu")
        model = PPO.load("highway_ppo/model", device=device)
        env = gym.make("highway-v0", render_mode='rgb_array',config={'screen_width':960, 'screen_height':120})

        with open('highway_ppo/network_hierarchy.txt', 'w') as file:
            file.write("-------------------------- Policy network  ---------------------------------\n")
            write_module_hierarchy_to_file(model.policy, file)
            file.write("-------------------------- Value function ----------------------------------\n")
            write_module_hierarchy_to_file(model.policy.value_net, file)

        for _ in range(50):
            obs, info = env.reset()
            done = truncated = False
            while not (done or truncated):
                action, _ = model.predict(obs)
                obs, reward, done, truncated, info = env.step(action)
                env.render()
