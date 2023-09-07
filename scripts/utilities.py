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
import os, shutil
from attention_network import EgoAttentionNetwork
import gymnasium as gym
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
import zipfile
import io
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler, Subset

from highway_env.envs.common.action import DiscreteMetaAction
ACTIONS_ALL = DiscreteMetaAction.ACTIONS_ALL

def clear_and_makedirs(directory):
    # Clear the directory to remove existing files
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

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

def DefaultActorCriticPolicy(env, device, **policy_kwargs):
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
                                    lr_schedule=lr_schedule,
                                    **policy_kwargs
                                  )

        # import torch
        # policy_net =  torch.nn.Sequential(
        #                                     torch.nn.Linear(state_dim, 64),
        #                                     torch.nn.LeakyReLU(),
        #                                     torch.nn.Dropout(0.2),
        #                                  ).to(device)
        # policy.mlp_extractor.policy_net = policy_net
        # action_net =  torch.nn.Sequential(
        #                                     torch.nn.Linear(64, 50),
        #                                     torch.nn.Tanh(),
        #                                     torch.nn.Dropout(0.3),
        #                                     torch.nn.Linear(50, action_dim),
        #                                  ).to(device)
        # policy.action_net =     action_net
        return policy   

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
        # print("data length for custom data set ",id(self), " is ", len(self.exp_acts),  len(self.exp_obs))
        return len(self.exp_acts)

    # def _load_episode_group(self):
    #     extract_expert_data(self.data_file, self.exp_obs, self.exp_acts, self.exp_dones, self.grp_idx)

    def _get_total_episode_groups(self):
        with h5py.File(self.data_file, 'r') as hf:
            total_episode_groups = len(list(hf.keys()))
        return total_episode_groups


    def __getitem__(self, idx):
        try:
            observation = torch.tensor(self.exp_obs[idx], dtype=torch.float32)
            action = torch.tensor(self.exp_acts[idx], dtype=torch.float32)
            done = torch.tensor(self.exp_dones[idx], dtype=torch.float32)
        except Exception as e:
            print(e , "for ", id(self), len(self.exp_obs), len(self.exp_acts), len(self.exp_dones))
            raise e

        # print(" Inside custom data set. Devices are ",self.device, observation.device, action.device, done.device)
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
    observer = env.unwrapped.observation_type
    obs = observer.observe()
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
        env = env.unwrapped
        if not obs_type.absolute and v_index > 0:
            v_position += env.vehicle.position # This is ego
        close_vehicles =       env.road.close_vehicles_to(env.vehicle,
                                                          env.PERCEPTION_DISTANCE,
                                                         count=observer.vehicles_count - 1,
                                                         see_behind=observer.see_behind,
                                                         sort=observer.order == "sorted")        
        vehicle = min(close_vehicles, key=lambda v: np.linalg.norm(v.position - v_position))
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



def save_checkpoint(project, run_name, epoch, trainer, metrics_plot_path):

    with wandb.init(
                        project=project, 
                        magic=True,
                    ) as run:
                    # if epoch is None:
                    epoch = "final"
                    run.log({f"metrics_plot": wandb.Image(metrics_plot_path)})
                    run.name = run_name
                    # Log the model as an artifact in wandb
                    torch.save(trainer , f"models_archive/BC_agent_{epoch}.pth") 
                    artifact = wandb.Artifact("trained_model_directory", type="model_directory")
                    artifact.add_dir("models_archive")
                    run.log_artifact(artifact)
    wandb.finish()
    clear_and_makedirs("models_archive")


class DownSamplingSampler(SubsetRandomSampler):
    def __init__(self, labels, class_weights, num_samples):
        """
        Args:
            class_weights (list): List of class weights.
            num_samples (int): Total number of samples to keep.
            seed (int): Seed for the random number generator.
        """
        self.class_weights = class_weights
        self.num_samples = num_samples
        # self.generator = torch.Generator()
        # self.generator.manual_seed(seed)
        self.class_labels = labels
        self.unique_labels = np.unique(self.class_labels)
        self.num_samples_per_class = int(num_samples/ len(self.unique_labels))
        self.indices = self._select_samples()

    def _select_samples(self):
        # Calculate the downsampled indices for each class
        self.downsampled_indices = []
        for class_label in self.unique_labels:
            class_indices = np.where(self.class_labels == class_label)[0]
            max_samples = min(len(class_indices), self.num_samples_per_class)
            downsampled_indices = class_indices[:max_samples]
            self.downsampled_indices.append(downsampled_indices)

        # Combine the downsampled indices for all classes
        return np.concatenate(self.downsampled_indices)
    
    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

def create_dataloaders(zip_filename, extract_path, device, **kwargs):

    # Create the extract_path if it doesn't exist
    if os.path.exists(extract_path):
        shutil.rmtree(extract_path)
    os.makedirs(extract_path)
    # Extract the HDF5 files from the zip archive
    # These files may be alredy existing because of a previous post process step.
    with zipfile.ZipFile(zip_filename, 'r') as archive:
        archive.extractall(extract_path)

    # Extract the names of the HDF5 files from the zip archive
    with zipfile.ZipFile(zip_filename, 'r') as archive:
        hdf5_train_file_names = [os.path.join(extract_path, name) 
                                    for name in archive.namelist() 
                                    if name.endswith('.h5') and "train" in name]
        hdf5_val_file_names = [os.path.join(extract_path, name) 
                                    for name in archive.namelist() 
                                    if name.endswith('.h5') and "val" in name]            
        # Create separate datasets for each HDF5 file
    train_datasets = [CustomDataset(hdf5_name, device) for hdf5_name in hdf5_train_file_names]


    # Create a combined dataset from the individual datasets
    combined_train_dataset = ConcatDataset(train_datasets)
    # Create shuffled indices for the combined dataset
    shuffled_indices = np.arange(len(combined_train_dataset))
    np.random.shuffle(shuffled_indices)

    # Create a shuffled version of the combined dataset using Subset
    shuffled_combined_train_dataset = Subset(combined_train_dataset, shuffled_indices)

    # Calculate the class frequencies
    all_actions = [sample['acts'] for sample in combined_train_dataset]
    action_frequencies = np.bincount(all_actions)
    class_weights = 1.0 / np.sqrt(action_frequencies)
    # class_weights =  np.array([np.exp(-freq/action_frequencies.sum()) for freq in action_frequencies])
    class_weights = class_weights / class_weights.sum()
    print(" class_weights at the end ", class_weights, " action_frequencies ", action_frequencies)

    # Calculate the least represented count
    least_represented_count = np.min(action_frequencies)

    # Get the number of unique action types
    num_action_types = len(np.unique(all_actions))

    num_samples=int(least_represented_count * num_action_types )
    desired_num_samples = 10000  # Adjust this value as needed
    seed = 42
    # sampler = DownSamplingSampler(
    #                                 labels = all_actions,
    #                                 class_weights = class_weights, 
    #                                 num_samples= num_samples
    #                              )
    print(" class_weights ", class_weights, " num_samples ", num_samples, " original samples fraction ", num_samples/len(all_actions))
    train_data_loader = DataLoader(
                                        shuffled_combined_train_dataset, 
                                        batch_size=kwargs['batch_size'], 
                                        # shuffle=True,
                                        # sampler=sampler,
                                        drop_last=True,
                                        num_workers=kwargs['n_cpu'],
                                        # pin_memory=True,
                                        # pin_memory_device=device,
                                 ) 
    return train_data_loader


def calculate_validation_metrics(bc_trainer,zip_filename, **training_kwargs):
    true_labels = []
    predicted_labels = []
    # Iterate through the validation data and make predictions
    with torch.no_grad():
        with zipfile.ZipFile(zip_filename, 'r') as zipf:
            hdf5_val_file_names = [file_name for file_name in zipf.namelist() if file_name.endswith('.h5') and "val" in file_name]
            for val_data_file in hdf5_val_file_names:
                with zipf.open(val_data_file) as file_in_zip:
                    hdf5_data = file_in_zip.read()
                    in_memory_file = io.BytesIO(hdf5_data)
                    val_obs, val_acts, val_dones = extract_post_processed_expert_data(in_memory_file)
                    predicted_labels.extend([bc_trainer.policy.predict(obs)[0] for obs in val_obs])
                    true_labels.extend(val_acts)

    # Calculate evaluation metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average=None)
    recall = recall_score(true_labels, predicted_labels, average=None)
    f1 = f1_score(true_labels, predicted_labels, average=None)
    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    # Print the metrics
    print("Accuracy:", accuracy, np.mean(accuracy))
    print("Precision:", precision, np.mean(precision))
    print("Recall:", recall, np.mean(recall))
    print("F1 Score:", f1, np.mean(f1))


    # predicted_labels = []
    # true_labels = []
    # with torch.no_grad():
    #     with zipfile.ZipFile(zip_filename, 'r') as zipf:
    #         hdf5_val_file_names = [file_name for file_name in zipf.namelist() if file_name.endswith('.h5') and "train" in file_name]
    #         for val_data_file in hdf5_val_file_names:
    #             with zipf.open(val_data_file) as file_in_zip:
    #                 hdf5_data = file_in_zip.read()
    #                 in_memory_file = io.BytesIO(hdf5_data)
    #                 val_obs, val_acts, val_dones = extract_post_processed_expert_data(in_memory_file)
    #                 predicted_labels.extend([bc_trainer.policy.predict(obs)[0] for obs in val_obs])
    #                 true_labels.extend(val_acts)

    # # Calculate evaluation metrics for training
    # tr_accuracy = accuracy_score(true_labels, predicted_labels)
    # tr_precision = precision_score(true_labels, predicted_labels, average=None)
    # tr_recall = recall_score(true_labels, predicted_labels, average=None)
    # tr_f1 = f1_score(true_labels, predicted_labels, average=None)



    # print("--------  Training data metrics for reference---------------")
    # print("Accuracy:", accuracy, np.mean(tr_accuracy))
    # print("Precision:", precision,  np.mean(tr_precision))
    # print("Recall:", recall, np.mean(tr_recall))
    # print("F1 Score:", f1, np.mean(tr_f1))


    plt.figure(figsize=(8, 6))
    class_labels = [ ACTIONS_ALL[idx] for idx in range(len(ACTIONS_ALL))]
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    heatmap_png = 'heatmap.png'
    plt.savefig(heatmap_png)

    if False: # For now keep this local as the remote artifact size is growing too much
        with zipfile.ZipFile(zip_filename, 'a') as zipf:
            zipf.write(heatmap_png, arcname=training_kwargs['plot_path'])
    # plt.show()  
    # print("saved confusion matrix")
    return accuracy, precision, recall, f1





