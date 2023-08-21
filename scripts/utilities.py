from copy import deepcopy as dcp
import torch.nn as nn
import h5py
import sys
from stable_baselines3.common.policies import ActorCriticPolicy
from models.nets import PolicyNetwork
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
from generate_expert_data import extract_post_processed_expert_data  
import pygame  
import numpy as np
import seaborn as sns
from highway_env.utils import lmap
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import wandb
import os
from attention_network import EgoAttentionNetwork
import gymnasium as gym

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

def DefaultActorCriticPolicy(env, device):
        state_dim = env.observation_space.high.shape[0]*env.observation_space.high.shape[1]
        action_dim = env.action_space.n
        def linear_decay_lr_schedule(step_num, initial_learning_rate, decay_steps, end_learning_rate):
            progress = step_num / decay_steps
            learning_rate = initial_learning_rate * (1 - progress) + end_learning_rate * progress
            return learning_rate
        
        # Create a callable learning rate schedule
        lr_schedule = lambda step_num: linear_decay_lr_schedule(
                                                                step_num, 
                                                                initial_learning_rate=0.001, 
                                                                decay_steps=100000, 
                                                                end_learning_rate=0.0001
                                                               )
        policy = ActorCriticPolicy(
                                    observation_space=env.observation_space,
                                    action_space=env.action_space,
                                    lr_schedule=lr_schedule
                                  )
        import torch
        policy.net_arch =     PolicyNetwork(
                                            state_dim=state_dim, 
                                            action_dim=action_dim, 
                                            discrete=True, 
                                            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
                                           )    

class CustomDataset(Dataset):
    def __init__(self, data_file, device, pad_value=0):
        # Load your data from the file and prepare it here
        # self.data = ...  # Load your data into this variable
        self.exp_obs, self.exp_acts, self.exp_dones = extract_post_processed_expert_data(data_file)
        self.pad_value = pad_value
        self.device = device
        # print(" data lengths ", len(self.exp_obs), len(self.exp_acts), len(self.exp_dones))
        return

    def __len__(self):
        return len(self.exp_acts)

    # def _load_episode_group(self):
    #     extract_expert_data(self.data_file, self.exp_obs, self.exp_acts, self.exp_dones, self.grp_idx)

    def _get_total_episode_groups(self):
        with h5py.File(self.data_file, 'r') as hf:
            total_episode_groups = len(list(hf.keys()))
        return total_episode_groups


    def __getitem__(self, idx):
        observation = torch.tensor(self.exp_obs[idx], dtype=torch.float32)
        action = torch.tensor(self.exp_acts[idx], dtype=torch.float32)
        done = torch.tensor(self.exp_dones[idx], dtype=torch.float32)

        sample = {
            'obs': observation,
            'acts': action,
            'dones' :done
        }
        return sample
        
    # def collate_fn(self, batch):
    #     observations = [sample['obs'] for sample in batch]
    #     actions = [sample['acts'] for sample in batch]
    #     dones = [sample['dones'] for sample in batch]
        
    #     # Pad sequences to the maximum length in the batch
    #     observations_padded = pad_sequence(observations, batch_first=True, padding_value=self.pad_value)
    #     actions_padded = pad_sequence(actions, batch_first=True, padding_value=self.pad_value)
    #     dones_padded = pad_sequence(dones, batch_first=True, padding_value=self.pad_value)
        
    #     return observations_padded, actions_padded, dones_padded



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

class CustomExtractor(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, **kwargs):
        super().__init__(observation_space, features_dim=kwargs["attention_layer_kwargs"]["feature_size"])
        self.extractor = EgoAttentionNetwork(**kwargs)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.extractor(observations)





def retrieve_gail_agents( artifact_version, project = None):
    # Initialize wandb
    wandb.init(project=project, name="inference")
    # Access the run containing the logged artifact

    # Download the artifact
    artifact = wandb.use_artifact(artifact_version)
    artifact_dir = artifact.download()
    wandb.finish()

    # Load the model from the downloaded artifact
    optimal_gail_agent_path = os.path.join(artifact_dir, "optimal_gail_agent.pth")
    final_gail_agent_path = os.path.join(artifact_dir, "final_gail_agent.pth")

    final_gail_agent = torch.load(final_gail_agent_path)
    optimal_gail_agent = torch.load(optimal_gail_agent_path)
    return optimal_gail_agent, final_gail_agent


