from copy import deepcopy as dcp
import torch.nn as nn
import h5py
import sys
from stable_baselines3.common.policies import ActorCriticPolicy
from models.nets import PolicyNetwork
    
def extract_expert_data(filename):
    exp_obs  = []
    exp_acts = []
    exp_done = []
    with h5py.File(filename, 'r') as hf:
        # List all the episode groups in the HDF5 file
        episode_groups = list(hf.keys())

        # Iterate through each episode group
        for episode_name in episode_groups:
            episode = hf[episode_name]

            # ep_obs  = []
            # ep_acts = []
            # ep_done = []

            # List all datasets (exp_obs and exp_acts) in the episode group
            datasets = list(episode.keys())

            # Iterate through each dataset in the episode group
            for dataset_name in datasets:
                dataset = episode[dataset_name]

                # Append the data to the corresponding list
                if dataset_name.startswith('exp_obs'):
                    exp_obs.extend([dataset[:]])
                elif dataset_name.startswith('exp_acts'):
                    exp_acts.extend([dataset[()]])
                elif dataset_name.startswith('exp_done'):
                    exp_done.extend([dataset[()]]) 
           

    return  exp_obs, exp_acts, exp_done


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