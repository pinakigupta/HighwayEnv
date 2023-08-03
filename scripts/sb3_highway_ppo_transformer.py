import functools
import gymnasium as gym
import pygame
import seaborn as sns
import torch as th
from highway_env.utils import lmap
from stable_baselines3 import PPO
from torch.distributions import Categorical
import torch
import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv
from tensorboard import program
import os, sys
import multiprocessing
from enum import Enum
import json
import copy
import pandas as pd
import wandb
from datetime import datetime
from torch import FloatTensor
from torch import nn
import h5py
import shutil
from models.nets import Expert
from models.gail import GAIL
from models.generate_expert_data import collect_expert_data

from sb3_callbacks import CustomCheckpointCallback, CustomMetricsCallback, CustomCurriculamCallback
from attention_network import EgoAttentionNetwork
# from utils import write_module_hierarchy_to_file

import warnings
warnings.filterwarnings("ignore")

class TrainEnum(Enum):
    RLTRAIN = 0
    RLDEPLOY = 1
    IRLTRAIN = 2
    IRLDEPLOY = 3
    EXPERT_DATA_COLLECTION =4



def append_key_to_dict_of_dict(kwargs, outer_key, inner_key, value):
    kwargs[outer_key] = {**kwargs.get(outer_key, {}), inner_key: value}

attention_network_kwargs = dict(
    # in_size=5*15,
    embedding_layer_kwargs={"in_size": 7, "layer_sizes": [64, 64], "reshape": False},
    attention_layer_kwargs={"feature_size": 64, "heads": 2},
    # num_layers = 3,
)

def write_module_hierarchy_to_file(model, file):
    def write_module_recursive(module, file=None, indent='', processed_submodules=None):
        if file is None:
            file = sys.stdout
        if processed_submodules is None:
            processed_submodules = set()

        num_members = [tuple(_.shape) for _ in module.parameters()]
        # num_members = len(list(module.modules())) - 1
        module_name = f'{module.__class__.__name__} (ID: {id(module)})'
        file.write(f'{indent}├─{module_name} '+ ' containing '+ str(len(num_members))  + ' items\n')

        if isinstance(module, nn.Sequential):
            for submodule in module:
                write_module_recursive(submodule, file, indent + '    ')
        elif isinstance(module, nn.ModuleList):
            for idx, submodule in enumerate(module):
                file.write(f'{indent}    ├─ModuleList[{idx}]\n')
                write_module_recursive(submodule, file, indent + '        ')
        else:
            for name, submodule in module._modules.items():
                if submodule not in processed_submodules:
                    processed_submodules.add(submodule)
                    write_module_recursive(submodule, file, indent + '    ')

            for name, submodule in module._parameters.items():
                if submodule is not None:
                    if submodule not in processed_submodules:
                        processed_submodules.add(submodule)
                        file.write(f'{indent}    ├─{name}: {submodule.shape}\n')

    write_module_recursive(model, file, processed_submodules=set())


class CustomExtractor(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, **kwargs):
        super().__init__(observation_space, features_dim=kwargs["attention_layer_kwargs"]["feature_size"])
        self.extractor = EgoAttentionNetwork(**kwargs)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.extractor(observations)

# ==================================
#     Environment configuration
# ==================================

def make_configure_env(**kwargs):
    env = gym.make(kwargs["id"], render_mode=kwargs["render_mode"], config=kwargs["config"])
    # env.configure(kwargs["config"])
    env.reset()
    return env


env_kwargs = {
    'id': 'highway-v0',
    'render_mode': 'rgb_array',
    'config': {
        "lanes_count": 4,
        "vehicles_count": 'random',
        "action": {
                "type": "DiscreteMetaAction",
            },
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 10,
            "features": [
                "presence",
                "x",
                "y",
                "vx",
                "vy",
                "cos_h",
                "sin_h"
            ],
            "absolute": False
        },
        "policy_frequency": 2,
        "duration": 40,
        "screen_width": 960,
        "screen_height": 180,
    }
}


# ==================================
#        Display attention matrix
# ==================================

def display_vehicles_attention(agent_surface, sim_surface, env, fe, min_attention=0.01):
        v_attention = compute_vehicles_attention(env, fe)
        # print("v_attention ", v_attention)
        # Extract the subsurface of the larger rectangle
        attention_surface = pygame.Surface(sim_surface.get_size(), pygame.SRCALPHA)
        pygame.draw.circle(
                                        surface=attention_surface,
                                        color=pygame.Color("white"),
                                        center=sim_surface.vec2pix(env.vehicle.position),
                                        radius=20,
                                        width=2
                          )
        for head in range(list(v_attention.values())[0].shape[0]):
            
            for vehicle, attention in v_attention.items():
                if attention[head] < min_attention:
                    continue
                # if True: 
                #     print("attention[head] ", attention[head], "vehicle ", vehicle)
                width = attention[head] * 5
                desat = np.clip(lmap(attention[head], (0, 0.5), (0.7, 1)), 0.7, 1)
                colors = sns.color_palette("dark", desat=desat)
                color = np.array(colors[(2*head) % (len(colors) - 1)]) * 255
                color = (*color, np.clip(lmap(attention[head], (0, 0.5), (100, 200)), 100, 200))
                pygame.draw.line(attention_surface, color,
                                     sim_surface.vec2pix(env.vehicle.position),
                                     sim_surface.vec2pix(vehicle.position),
                                     max(sim_surface.pix(width), 1)
                                )
            # subsurface = attention_surface.subsurface(pygame.Rect(0, 0, 4800, 200))
            sim_surface.blit(attention_surface, (0, 0))

def compute_vehicles_attention(env,fe):
    obs = env.unwrapped.observation_type.observe()
    obs_t = torch.tensor(obs[None, ...], dtype=torch.float)
    attention = fe.extractor.get_attention_matrix(obs_t)
    attention = attention.squeeze(0).squeeze(1).detach().cpu().numpy()
    ego, others, mask = fe.extractor.split_input(obs_t)
    mask = mask.squeeze()
    v_attention = {}
    obs_type = env.observation_type
    if hasattr(obs_type, "agents_observation_types"):  # Handle multi-model observation
        obs_type = obs_type.agents_observation_types[0]
    for v_index in range(obs.shape[0]):
        if mask[v_index]:
            continue
        v_position = {}
        for feature in ["x", "y"]:
            v_feature = obs[v_index, obs_type.features.index(feature)]
            v_feature = lmap(v_feature, [-1, 1], obs_type.features_range[feature])
            v_position[feature] = v_feature
        v_position = np.array([v_position["x"], v_position["y"]])
        if not obs_type.absolute and v_index > 0:
            v_position += env.unwrapped.vehicle.position # This is ego
        vehicle = min(env.unwrapped.road.vehicles, key=lambda v: np.linalg.norm(v.position - v_position))
        v_attention[vehicle] = attention[:, v_index]
    return v_attention

# ==================================
#        Main script  20 
# ==================================

if __name__ == "__main__":
    train = TrainEnum.RLTRAIN
    policy_kwargs = dict(
            features_extractor_class=CustomExtractor,
            features_extractor_kwargs=attention_network_kwargs,
        )
    
    WARM_START = False
    # Get the current date and time
    now = datetime.now()
    month = now.strftime("%m")
    day = now.strftime("%d")
    expert_data='expert_data.h5'
    n_cpu =  multiprocessing.cpu_count()

    def timenow():
        return now.strftime("%H%M")

    
    if   train == TrainEnum.EXPERT_DATA_COLLECTION: # EXPERT_DATA_COLLECTION
        # env = make_configure_env(**env_kwargs).unwrapped
        with open("config.json") as f:
            config = json.load(f)
        device = torch.device("cpu")
        expert = PPO.load("highway_attention_ppo/model", device=device) # This is not really ultimately treated as expert. Just some policy to run ego.
        append_key_to_dict_of_dict(env_kwargs,'config','mode','expert')
        exp_rwd_iter, exp_obs, exp_acts   =           collect_expert_data  (
                                                                                expert,
                                                                                config["num_expert_steps"],
                                                                                filename=expert_data,
                                                                                **env_kwargs
                                                                            )
    elif train == TrainEnum.RLTRAIN: # training 
        append_key_to_dict_of_dict(env_kwargs,'config','duration',20)
        env = make_vec_env(make_configure_env, n_envs=n_cpu, vec_env_cls=SubprocVecEnv, env_kwargs=env_kwargs)
        total_timesteps=200*1000
        # Set the checkpoint frequency
        checkpoint_freq = total_timesteps/1000  # Save the model every 10,000 timesteps
        model = PPO(
                    "MlpPolicy", 
                    env,
                    n_steps=512 // n_cpu,
                    batch_size=64,
                    learning_rate=2e-3,
                    policy_kwargs=policy_kwargs,
                    verbose=1,
                    )
        

        checkptcallback = CustomCheckpointCallback(checkpoint_freq, 'checkpoint')  # Create an instance of the custom callback
        # Train the model
        with wandb.init(
                        project="RL", 
                        magic=True,
                        ) as run:
            run.name = f"sweep_{month}{day}_{timenow()}"
            # Create the custom callback
            metrics_callback = CustomMetricsCallback()
            curriculamcallback = CustomCurriculamCallback()

            if WARM_START:
                # Download the artifact in case you want to initialize with pre - trained 
                artifact = wandb.use_artifact("trained_model:v6")
                artifact_dir = artifact.download()

                # Load the model from the downloaded artifact
                rl_agent_path = os.path.join(artifact_dir, "RL_agent.pth")
                # if rl_agent_path in locals():
                model.policy.load_state_dict(torch.load(rl_agent_path))

            training_info = model.learn(
                                        total_timesteps=total_timesteps,
                                        callback=[
                                                    checkptcallback, 
                                                    curriculamcallback,
                                                 ]
                                        )
            
            # Log the model as an artifact in wandb
            torch.save(model.policy.state_dict(), 'RL_agent.pth')
            
            artifact = wandb.Artifact("trained_model", type="model")
            artifact.add_file("RL_agent.pth")
            run.log_artifact(artifact)

        wandb.finish()
        model.save("highway_attention_ppo/model_new")

        # Save the final model
        # model.save("highway_attention_ppo/model")
    elif train==TrainEnum.IRLTRAIN:
        
        sweep_config = {
            "method": "grid",
            "metric": {
                "name": "episode_reward",
                "goal": "maximize"
            },
            "parameters": {
                "duration": {
                    "values": [500]  # Values for the "duration" field to be swept
                }
            }
        }
        project_name = f"random_env_gail"
        sweep_id = wandb.sweep(sweep_config, project=project_name)
        device = torch.device("cpu")
        
        # IDM + MOBIL is treated as expert.
        with open("config.json") as f:
            config = json.load(f)

        config_defaults = {
            "duration": 400  # Default value for the "duration" field
        }
        # Customize the project name with the current date and time
        exp_obs  = []
        exp_acts = []
        with h5py.File('expert_data.h5', 'r') as hf:
            # List all the episode groups in the HDF5 file
            episode_groups = list(hf.keys())

            # Iterate through each episode group
            for episode_name in episode_groups:
                episode = hf[episode_name]

                # List all datasets (exp_obs and exp_acts) in the episode group
                datasets = list(episode.keys())

                # Iterate through each dataset in the episode group
                for dataset_name in datasets:
                    dataset = episode[dataset_name]

                    # Append the data to the corresponding list
                    if dataset_name.startswith('exp_obs'):
                        exp_obs.append(dataset[:])
                    elif dataset_name.startswith('exp_acts'):
                        exp_acts.append(dataset[()])

            # for i in range(len(hf.keys()) // 2):
            #     exp_obs.append(hf[f'exp_obs{i}'][:])
            #     exp_acts.append(hf[f'exp_acts{i}'][()])

        import random
        # exp_obs, exp_acts = list(zip(*random.sample(list(zip(exp_obs, exp_acts)), len(exp_acts))))
        exp_obs = FloatTensor(exp_obs)
        exp_acts = FloatTensor(exp_acts)

        def train_gail_agent(exp_obs=exp_obs, exp_acts=exp_acts, gail_agent_path = None, **env_kwargs):
            env= make_configure_env(**env_kwargs).unwrapped
            state_dim = env.observation_space.high.shape[0]*env.observation_space.high.shape[1]
            action_dim = env.action_space.n
            gail_agent = GAIL(
                                state_dim, 
                                action_dim , 
                                discrete=True, 
                                device=torch.device("cpu"), 
                                **config, 
                                # **policy_kwargs, 
                                observation_space= env.observation_space
                             ).to(device=device)
            if gail_agent_path is not None:
                gail_agent.load_state_dict(torch.load(gail_agent_path))
            rewards, optimal_agent = gail_agent.train(exp_obs=exp_obs, exp_acts=exp_acts, **env_kwargs)
            return rewards, optimal_agent

        def train_sweep(exp_obs, exp_acts):
            with wandb.init(
                            project=project_name, 
                            config=config_defaults,
                            magic=True,
                           ) as run:
                run.name = f"sweep_{month}{day}_{timenow()}"

                gail_agent_path = None
                if WARM_START:
                    # Download the artifact in case you want to initialize with pre - trained 
                    artifact = wandb.use_artifact("trained_model:v7")
                    artifact_dir = artifact.download()
                    # Load the model from the downloaded artifact
                    gail_agent_path = os.path.join(artifact_dir, "optimal_gail_agent.pth")
                    

                rewards, optimal_agent = train_gail_agent(
                                                            exp_obs=exp_obs, 
                                                            exp_acts=exp_acts, 
                                                            gail_agent_path= gail_agent_path, 
                                                            **env_kwargs
                                                         )

                # Log the reward vector for each epoch
                for epoch, reward in enumerate(rewards, 1):
                    run.log({f"epoch_{epoch}_rewards": reward})
                
                # Create a directory for the models
                os.makedirs("models_archive", exist_ok=True)

                shutil.move("optimal_gail_agent.pth", "models_archive/optimal_gail_agent.pth")
                shutil.move("final_gail_agent.pth", "models_archive/final_gail_agent.pth")

                # Log the model as an artifact in wandb
                artifact = wandb.Artifact("trained_model_directory", type="model_directory")
                artifact.add_dir("models_archive")
                run.log_artifact(artifact)

                    
        # rewards, optimal_agent = train_gail_agent(exp_obs=exp_obs, exp_acts=exp_acts, **env_kwargs)

        wandb.agent(
                     sweep_id=sweep_id, 
                     function=lambda: train_sweep(exp_obs, exp_acts)
                   )
        wandb.finish()
    elif train==TrainEnum.IRLDEPLOY:
        # Set the WANDB_MODE environment variable
        # os.environ["WANDB_MODE"] = "offline"
        # Initialize wandb
        wandb.init(project="random_env_gail", name="inference")
        # Access the run containing the logged artifact

        # Download the artifact
        artifact = wandb.use_artifact("trained_model:latest")
        artifact_dir = artifact.download()

        # Load the model from the downloaded artifact
        gail_agent_path = os.path.join(artifact_dir, "optimal_gail_agent.pth")


        env = make_configure_env(**env_kwargs,duration=400)
        state_dim = env.observation_space.high.shape[0]*env.observation_space.high.shape[1]
        action_dim = env.action_space.n
        with open("config.json") as f:
            config = copy.deepcopy(json.load(f))
        # Load the GAIL model
        loaded_gail_agent = GAIL(
                                 state_dim, 
                                 action_dim, 
                                 discrete=True, device=torch.device("cpu"), 
                                 **config, 
                                #  **policy_kwargs, 
                                 observation_space= env.observation_space
                                 )
        loaded_gail_agent.load_state_dict(torch.load(gail_agent_path))
        # loaded_gail_agent.eval()
        wandb.finish()

        env.render()
        gamma = 1.0
        # env.viewer.set_agent_display(functools.partial(display_vehicles_attention, env=env, 
        #                                                fe=loaded_gail_agent.features_extractor))
        for _ in range(500):
            obs, info = env.reset()
            done = truncated = False
            cumulative_reward = 0
            while not (done or truncated):
                action = loaded_gail_agent.act(obs)
                # print("action : " , action)
                # print("obs : ", obs.flatten())
                obs, reward, done, truncated, info = env.step(action)
                cumulative_reward += gamma * reward
                # print("speed: ",env.vehicle.speed," ,reward: ", reward, " ,cumulative_reward: ",cumulative_reward)
                env.render()
            print("--------------------------------------------------------------------------------------")
    elif train==TrainEnum.RLDEPLOY:
        env = make_configure_env(**env_kwargs,duration=400)
        device = torch.device("cpu")
        model = PPO(
                    "MlpPolicy", 
                    env,
                    # n_steps=512 // n_cpu,
                    batch_size=64,
                    learning_rate=2e-3,
                    policy_kwargs=policy_kwargs,
                    device=device
                    )
        with open('highway_attention_ppo/network_hierarchy.txt', 'w') as file:
            file.write("-------------------------- Policy network  ---------------------------------\n")
            write_module_hierarchy_to_file(model.policy, file)
            file.write("-------------------------- Value function ----------------------------------\n")
            write_module_hierarchy_to_file(model.policy.value_net, file)
        
        wandb.init(project="RL", name="inference")
        # Access the run containing the logged artifact

        # Download the artifact
        artifact = wandb.use_artifact("trained_model:latest")
        artifact_dir = artifact.download()

        # Load the model from the downloaded artifact
        rl_agent_path = os.path.join(artifact_dir, "RL_agent.pth")
        model.policy.load_state_dict(torch.load(rl_agent_path))
        wandb.finish()
        
        env.render()
        gamma = 1.0
        env.viewer.set_agent_display(functools.partial(display_vehicles_attention, env=env, fe=model.policy.features_extractor))
        for _ in range(50):
            obs, info = env.reset()
            done = truncated = False
            cumulative_reward = 0
            while not (done or truncated):
                action, _ = model.predict(obs)
                obs, reward, done, truncated, info = env.step(action)
                cumulative_reward += gamma * reward
                print("speed: ",env.vehicle.speed," ,reward: ", reward, " ,cumulative_reward: ",cumulative_reward)
                env.render()
            print("--------------------------------------------------------------------------------------")
