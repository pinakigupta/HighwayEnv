import functools
import gymnasium as gym
import pygame
import seaborn as sns
import torch as th
from highway_env.utils import lmap
from stable_baselines3 import PPO
from torch.distributions import Categorical
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
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
import h5py
from models.nets import Expert
from models.gail import GAIL
from models.generate_expert_data import collect_expert_data

class TrainEnum(Enum):
    RLTRAIN = 0
    RLDEPLOY = 1
    IRLTRAIN = 2
    IRLDEPLOY = 3
    EXPERT_DATA_COLLECTION =4

from stable_baselines3.common.callbacks import BaseCallback

class CustomCheckpointCallback(BaseCallback):
    def __init__(self, save_freq, save_path):
        super(CustomCheckpointCallback, self).__init__()
        self.save_freq = save_freq
        self.save_path = save_path

    def _init_callback(self) -> None:
        if self.save_freq > 0:
            self.model.save(self.save_path)  # Save the initial model

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            self.model.save(self.save_path)  # Save the model at specified intervals
        return True

# ==================================
#        Policy Architecture
# ==================================

def activation_factory(activation_type):
    if activation_type == "RELU":
        return F.relu
    elif activation_type == "TANH":
        return torch.tanh
    elif activation_type == "ELU":
        return nn.ELU()
    else:
        raise ValueError("Unknown activation_type: {}".format(activation_type))

class BaseModule(torch.nn.Module):
    """
    Base torch.nn.Module implementing basic features:
        - initialization factory
        - normalization parameters
    """

    def __init__(self, activation_type="RELU", reset_type="XAVIER"):
        super().__init__()
        self.activation = activation_factory(activation_type)
        self.reset_type = reset_type

    def _init_weights(self, m):
        if hasattr(m, 'weight'):
            if self.reset_type == "XAVIER":
                torch.nn.init.xavier_uniform_(m.weight.data)
            elif self.reset_type == "ZEROS":
                torch.nn.init.constant_(m.weight.data, 0.)
            else:
                raise ValueError("Unknown reset type")
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.)

    def reset(self):
        self.apply(self._init_weights)


class MultiLayerPerceptron(BaseModule):
    def __init__(self,
                 in_size=None,
                 layer_sizes=None,
                 reshape=True,
                 out_size=None,
                 activation="RELU",
                 is_policy=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.reshape = reshape
        self.layer_sizes = layer_sizes or [64, 64]
        self.out_size = out_size
        self.activation = activation_factory(activation)
        self.is_policy = is_policy
        self.softmax = nn.Softmax(dim=-1)
        sizes = [in_size] + self.layer_sizes
        layers_list = [nn.Linear(sizes[i], sizes[i + 1])
                       for i in range(len(sizes) - 1)]
        self.layers = nn.ModuleList(layers_list)
        if out_size:
            self.predict = nn.Linear(sizes[-1], out_size)

    def forward(self, x):
        if self.reshape:
            x = x.reshape(x.shape[0], -1)  # We expect a batch of vectors
        for layer in self.layers:
            x = self.activation(layer(x.float()))
        if self.out_size:
            x = self.predict(x)
        if self.is_policy:
            action_probs = self.softmax(x)
            dist = Categorical(action_probs)
            return dist
        return x

    def action_scores(self, x):
        if self.is_policy:
            if self.reshape:
                x = x.reshape(x.shape[0], -1)  # We expect a batch of vectors
            for layer in self.layers:
                x = self.activation(layer(x.float()))
            if self.out_size:
                action_scores = self.predict(x)
            return action_scores


class EgoAttention(BaseModule):
    def __init__(self,
                 feature_size=64,
                 heads=4,
                 dropout_factor=0):
        super().__init__()
        self.feature_size = feature_size
        self.heads = heads
        self.dropout_factor = dropout_factor
        self.features_per_head = int(self.feature_size / self.heads)

        self.value_all = nn.Linear(self.feature_size,
                                   self.feature_size,
                                   bias=False)
        self.key_all = nn.Linear(self.feature_size,
                                 self.feature_size,
                                 bias=False)
        self.query_ego = nn.Linear(self.feature_size,
                                   self.feature_size,
                                   bias=False)
        self.attention_combine = nn.Linear(self.feature_size,
                                           self.feature_size,
                                           bias=False)

    @classmethod
    def default_config(cls):
        return {
        }

    def forward(self, ego, others, mask=None):
        batch_size = others.shape[0]
        n_entities = others.shape[1] + 1
        input_all = torch.cat((ego.view(batch_size, 1,
                               self.feature_size), others), dim=1)
        # Dimensions: Batch, entity, head, feature_per_head
        key_all = self.key_all(input_all).view(batch_size,
                                               n_entities,
                                               self.heads,
                                               self.features_per_head)
        value_all = self.value_all(input_all).view(batch_size,
                                                   n_entities,
                                                   self.heads,
                                                   self.features_per_head)
        query_ego = self.query_ego(ego).view(batch_size, 1,
                                             self.heads,
                                             self.features_per_head)

        # Dimensions: Batch, head, entity, feature_per_head
        key_all = key_all.permute(0, 2, 1, 3)
        value_all = value_all.permute(0, 2, 1, 3)
        query_ego = query_ego.permute(0, 2, 1, 3)
        if mask is not None:
            mask = mask.view((batch_size, 1, 1,
                              n_entities)).repeat((1, self.heads, 1, 1))
        value, attention_matrix = attention(query_ego,
                                            key_all,
                                            value_all,
                                            mask,
                                            nn.Dropout(self.dropout_factor))
        result = (self.attention_combine(
                    value.reshape((batch_size,
                                   self.feature_size))) + ego.squeeze(1))/2
        return result, attention_matrix


class EgoAttentionNetwork(BaseModule):
    def __init__(self,
                 in_size=None,
                 out_size=None,
                 presence_feature_idx=0,
                 embedding_layer_kwargs=None,
                 attention_layer_kwargs=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.out_size = out_size
        self.presence_feature_idx = presence_feature_idx
        embedding_layer_kwargs = embedding_layer_kwargs or {}
        if not embedding_layer_kwargs.get("in_size", None):
            embedding_layer_kwargs["in_size"] = in_size
        # if 'num_layers' in kwargs:
        #     self.encoder = nn.ModuleList([EncoderLayer(d_model, heads) for _ in range(kwargs['num_layers'])])
        self.ego_embedding = MultiLayerPerceptron(**embedding_layer_kwargs)
        self.embedding = MultiLayerPerceptron(**embedding_layer_kwargs)

        attention_layer_kwargs = attention_layer_kwargs or {}
        self.attention_layer = EgoAttention(**attention_layer_kwargs)

    def forward(self, x):
        ego_embedded_att, _ = self.forward_attention(x)
        # print(ego_embedded_att.shape)
        # for _ in range(4):
        #     ego_embedded_att, _=self.forward_attention(ego_embedded_att)
        return ego_embedded_att

    def split_input(self, x, mask=None):
        # Dims: batch, entities, features
        if len(x.shape) == 2:
            x = x.unsqueeze(axis=0)
        ego = x[:, 0:1, :]
        others = x[:, 1:, :]
        if mask is None:
            aux = self.presence_feature_idx
            mask = x[:, :, aux:aux + 1] < 0.5
        return ego, others, mask

    def forward_attention(self, x):
        ego, others, mask = self.split_input(x)
        ego = self.ego_embedding(ego)
        others = self.embedding(others)
        return self.attention_layer(ego, others, mask)

    def get_attention_matrix(self, x):
        _, attention_matrix = self.forward_attention(x)
        return attention_matrix


def attention(query, key, value, mask=None, dropout=None):
    """
    Compute a Scaled Dot Product Attention.

    Parameters
    ----------
    query
        size: batch, head, 1 (ego-entity), features
    key
        size: batch, head, entities, features
    value
        size: batch, head, entities, features
    mask
        size: batch,  head, 1 (absence feature), 1 (ego-entity)
    dropout

    Returns
    -------
    The attention softmax(QK^T/sqrt(dk))V
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    output = torch.matmul(p_attn, value)
    return output, p_attn


attention_network_kwargs = dict(
    # in_size=5*15,
    embedding_layer_kwargs={"in_size": 7, "layer_sizes": [64, 64], "reshape": False},
    attention_layer_kwargs={"feature_size": 64, "heads": 2},
    # num_layers = 3,
)


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
    env = gym.make(kwargs["id"], render_mode=kwargs["render_mode"])
    env.configure(kwargs["config"])
    env.reset()
    return env


env_kwargs = {
    'id': 'highway-v0',
    'render_mode': 'rgb_array',
    'config': {
        "mode" : 'expert',
        "lanes_count": 4,
        "vehicles_count": 50,
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

# ==================================
#        Main script  20 
# ==================================

if __name__ == "__main__":
    train = TrainEnum.IRLTRAIN
    policy_kwargs = dict(
            features_extractor_class=CustomExtractor,
            features_extractor_kwargs=attention_network_kwargs,
        )
    
    # Get the current date and time
    now = datetime.now()
    month = now.strftime("%m")
    day = now.strftime("%d")
    expert_data='expert_data.h5'

    def timenow():
        return now.strftime("%H%M")

    
    if   train == TrainEnum.EXPERT_DATA_COLLECTION: # EXPERT_DATA_COLLECTION
        env = make_configure_env(**copy.deepcopy(env_kwargs)).unwrapped
        with open("config.json") as f:
            config = json.load(f)
        device = torch.device("cpu")
        expert = PPO.load("checkpoint", device=device) # This is not really ultimately treated as expert. Just some policy to run ego.
        exp_rwd_iter, exp_obs, exp_acts   =           collect_expert_data  (
                                                                                env,
                                                                                expert,
                                                                                config["num_expert_steps"],
                                                                                filename=expert_data
                                                                            )
    elif train == TrainEnum.RLTRAIN: # training 
        n_cpu =  multiprocessing.cpu_count()
        env = make_vec_env(make_configure_env, n_envs=n_cpu, vec_env_cls=SubprocVecEnv, env_kwargs=env_kwargs)
        # Set the checkpoint frequency
        checkpoint_freq = 20000  # Save the model every 10,000 timesteps
        model = PPO(
                    "MlpPolicy", env,
                    n_steps=512 // n_cpu,
                    batch_size=64,
                    learning_rate=2e-3,
                    policy_kwargs=policy_kwargs,
                    verbose=2,
                    tensorboard_log="highway_attention_ppo/"
                    )
        callback = CustomCheckpointCallback(checkpoint_freq, 'checkpoint')  # Create an instance of the custom callback
        # Train the model
        model.learn(
                    total_timesteps=200*1000,
                    callback=callback
                    )
        # Save the final model
        model.save("highway_attention_ppo/model")
    elif train==TrainEnum.IRLTRAIN:
        
        sweep_config = {
            "method": "grid",
            "metric": {
                "name": "episode_reward",
                "goal": "maximize"
            },
            "parameters": {
                "duration": {
                    "values": [500, 1000, 2500]  # Values for the "duration" field to be swept
                }
            }
        }
        project_name = f"gail_hyperparameter_tuning_2"
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


        exp_obs = FloatTensor(exp_obs)
        exp_acts = FloatTensor(exp_acts)

        def train_sweep(exp_obs, exp_acts):
            with wandb.init(
                            project=project_name, 
                            config=config_defaults,
                            magic=True,
                           ) as run:
                run.name = f"sweep_{month}{day}_{timenow()}"
                env = make_configure_env(**copy.deepcopy(env_kwargs), mode="expert").unwrapped
                state_dim = env.observation_space.high.shape[0]*env.observation_space.high.shape[1]
                action_dim = env.action_space.n
                gail_agent = GAIL(state_dim, action_dim , discrete=True, device=torch.device("cpu"), 
                                **config, observation_space= env.observation_space).to(device=device)
                # expert = Expert(state_dim, action_dim, discrete=True, **expert_config).to(device)
                rewards, optimal_agent = gail_agent.train(env=env, exp_obs=exp_obs, exp_acts=exp_acts)

                # Log the reward vector for each epoch
                for epoch, reward in enumerate(rewards, 1):
                    run.log({f"epoch_{epoch}_rewards": reward})
                
                # Log the model as an artifact in wandb
                artifact = wandb.Artifact("trained_model", type="model")
                artifact.add_file("optimal_gail_agent.pth")
                run.log_artifact(artifact)

                    
        env = make_configure_env(**copy.deepcopy(env_kwargs)).unwrapped
        state_dim = env.observation_space.high.shape[0]*env.observation_space.high.shape[1]
        action_dim = env.action_space.n

        # gail_agent = GAIL(state_dim, action_dim , discrete=True, device=torch.device("cpu"), 
        #                                 **config, observation_space= env.observation_space).to(device=device)
        # rewards, optimal_agent = gail_agent.train(env=env, exp_obs=exp_obs, exp_acts=exp_acts)

        wandb.agent(
                     sweep_id=sweep_id, 
                     function=lambda: train_sweep(exp_obs, exp_acts)
                   )
        wandb.finish()
    elif train==TrainEnum.IRLDEPLOY:
        # Initialize wandb
        wandb.init(project="gail_hyperparameter_tuning_2")
        # Access the run containing the logged artifact
        run = wandb.init()

        # Download the artifact
        artifact = run.use_artifact("trained_model:v2")
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

        env.render()
        gamma = 1.0
        # env.viewer.set_agent_display(functools.partial(display_vehicles_attention, env=env, 
        #                                                fe=loaded_gail_agent.features_extractor))
        for _ in range(50):
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
        model = PPO.load("checkpoint", device=device)
        with open('highway_attention_ppo/network_hierarchy.txt', 'w') as file:
            file.write("-------------------------- Policy network  ---------------------------------\n")
            write_module_hierarchy_to_file(model.policy, file)
            file.write("-------------------------- Value function ----------------------------------\n")
            write_module_hierarchy_to_file(model.policy.value_net, file)


        # if os.path.exists(log_dir):
        #     print("Log directory exists.", log_dir)
        #     # Run TensorBoard command or perform other operations
        #     # Launch TensorBoard
        #     tb.main()
        # else:
        #     print("Log directory not found. Please check the path.", log_dir)
        

        
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
