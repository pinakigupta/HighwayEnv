from copy import deepcopy as dcp
import torch.nn as nn
import h5py
import sys
from stable_baselines3.common.policies import ActorCriticPolicy
from models.nets import PolicyNetwork
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
    
def extract_expert_data(filename, exp_obs = [], exp_acts = [], exp_dones = [], episode_index=None):
    if not exp_obs:
        exp_obs  = []
    if not exp_acts:
        exp_acts = []
    if not exp_dones:
        exp_dones = []
    with h5py.File(filename, 'r') as hf:
        # List all the episode groups in the HDF5 file
        episode_groups = list(hf.keys())

        if episode_index is not None:
            episode_name = episode_groups[episode_index]
            episode = hf[episode_name]

            for dataset_name, dataset in episode.items():
                if dataset_name.startswith('exp_obs'):
                    exp_obs.extend([dataset[:]])
                elif dataset_name.startswith('exp_acts'):
                    exp_acts.extend([dataset[()]])
                elif dataset_name.startswith('exp_done'):
                    exp_dones.extend([dataset[()]])
        else:
            for episode_name in episode_groups:
                episode = hf[episode_name]

                for dataset_name, dataset in episode.items():
                    if dataset_name.startswith('exp_obs'):
                        exp_obs.extend([dataset[:]])
                    elif dataset_name.startswith('exp_acts'):
                        exp_acts.extend([dataset[()]])
                    elif dataset_name.startswith('exp_done'):
                        exp_dones.extend([dataset[()]])
          

    return  exp_obs, exp_acts, exp_dones


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
        # self.exp_obs, self.exp_acts, self.exp_dones = extract_expert_data(data_file)
        self.data_file = data_file
        self.pad_value = pad_value
        self.device = device
        self.exp_obs = []
        self.exp_acts = []
        self.exp_dones = []
        self.grp_idx = 0  # Initialize episode group index
        self.total_episode_groups = self._get_total_episode_groups()
        print(" data lengths ", len(self.exp_obs), len(self.exp_acts), len(self.exp_dones))

    def __len__(self):
        # return len(self.exp_acts)
        return int(1000000)

    def _load_episode_group(self):
        extract_expert_data(self.data_file, self.exp_obs, self.exp_acts, self.exp_dones, self.grp_idx)

    def _get_total_episode_groups(self):
        with h5py.File(self.data_file, 'r') as hf:
            total_episode_groups = len(list(hf.keys()))
        return total_episode_groups


    def __getitem__(self, idx):
        while True:
            if idx >= len(self.exp_acts):
                self.grp_idx += 1
                if self.grp_idx >= self.total_episode_groups:
                    # No more data left, raise a StopIteration exception
                    raise StopIteration("No more data available.")

                self._load_episode_group()
            else:
                break

        # Return a dictionary with "obs" and "acts" as Tensors
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