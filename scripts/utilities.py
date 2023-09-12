from copy import deepcopy as dcp
import torch.nn as nn
from torch import multiprocessing
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
from tqdm import tqdm
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

def process_file(train_data_file, result_queue, device):
    processed_data = CustomDataset(train_data_file, device)
    result_queue.put(processed_data)

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

        return policy   

class CustomDataset(Dataset):
    def __init__(self, data_file, device, pad_value=0, **kwargs):
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



def display_vehicles_attention(agent_surface, sim_surface, env, fe, device,  min_attention=0.01):
        v_attention = compute_vehicles_attention(env, fe, device)
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

def compute_vehicles_attention(env,fe, device):
    observer = env.unwrapped.observation_type
    obs = observer.observe()
    obs_t = torch.tensor(obs[None, ...], dtype=torch.float).to(device)
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



def save_checkpoint(project, run_name, epoch, model, **kwargs):
    # if epoch is None:
    epoch = "final"
    model_path = f"models_archive/agent_{epoch}.pth"
    jit_model_path = f"models_archive/agent_{epoch}.pt"
    zip_filename = f"models_archive/agent.zip"
    torch.save( obj = model , f= model_path, pickle_protocol=5) 
    torch.jit.trace(torch.load(model_path), torch.randn(1, *model.observation_space.shape)).save(jit_model_path)
    os.remove(model_path)
    # with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
    #     # Add the file to the zip archive
    #     zipf.write(model_path)
    #     os.remove(model_path)
    with wandb.init(
                        project=project, 
                        magic=True,
                    ) as run:                    
                    if 'metrics_plot_path' in kwargs and kwargs['metrics_plot_path']:
                        run.log({f"metrics_plot": wandb.Image(kwargs['metrics_plot_path'])})
                    run.name = run_name
                    # Log the model as an artifact in wandb
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

def create_dataloaders(zip_filename, train_datasets, device, visited_data_files, **kwargs):

    # Create the extract_path if it doesn't exist
    # if os.path.exists(extract_path):
    #     shutil.rmtree(extract_path)
    # os.makedirs(extract_path)
    # Extract the HDF5 files from the zip archive
    # These files may be alredy existing because of a previous post process step.
    # with zipfile.ZipFile(zip_filename, 'r') as archive:
    #     archive.extractall(extract_path)

    result_queue = multiprocessing.Queue() 
    manager = multiprocessing.Manager()
    managed_visited_data_files = manager.list(list(visited_data_files))

    # Use an Event to signal when all processes have started
    # started_event = multiprocessing.Event()
    extract_path = 'data'
    # Extract the names of the HDF5 files from the zip archive
    with zipfile.ZipFile(zip_filename, 'r') as zipf:
        hdf5_train_file_names = [file_name for file_name in zipf.namelist() if file_name.endswith('.h5') and "train" in file_name]
        zipf.extractall(extract_path)
    worker_processes = []
    for train_file in hdf5_train_file_names:
        worker_process = multiprocessing.Process(
                                                    target= process_file, 
                                                    args=(
                                                            os.path.join(extract_path, train_file), result_queue, device
                                                          )
                                                )
        worker_processes.append(worker_process)
        worker_process.start()

        # Wait until all processes have started
        # started_event.wait()
    pbar_joining = tqdm(total=len(worker_processes), desc='Data loaders Finishing')
    while not result_queue.empty():
        processed_data = result_queue.get()
        train_datasets.extend([processed_data])

    for worker_process in worker_processes:
        worker_process.join()
        pbar_joining.update(1)

        # Create separate datasets for each HDF5 file
    # train_datasets = [CustomDataset(hdf5_name, device) for hdf5_name in hdf5_train_file_names]
    # visited_data_files = set(managed_visited_data_files)


    shutil.rmtree(extract_path)

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


def calculate_validation_metrics(policy,zip_filename, **training_kwargs):
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
                    in_memory_file.close()
                    predicted_labels.extend([policy.predict(obs)[0] for obs in val_obs])
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

    if False:
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


class CustomImageExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, hidden_dim=64):
        super(CustomImageExtractor, self).__init__(observation_space, hidden_dim)
        
        if not isinstance(observation_space, gym.spaces.Box):
            raise ValueError("Observation space must be continuous (e.g., image-based)")
        
        # Define a pretrained ResNet model as the feature extractor trunk
        self.resnet = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)
        self.resnet.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Adjust the last fully connected layer for feature dimension control
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, hidden_dim)
        
        # Additional fully connected layer for custom hidden feature dimension
        self.fc_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.height, self.width = observation_space.shape[1], observation_space.shape[2]
    
    def forward(self, observations):

        # Reshape observations to [batch_size, channels, height, width]
        observations = observations.view(observations.size(0), -1, self.height, self.width)
        # Normalize pixel values to the range [0, 1] (if necessary)
        observations = observations / 255.0

        # Forward pass through the ResNet trunk for feature extraction
        resnet_features = self.resnet(observations)
        
        # Apply a fully connected layer for the custom hidden feature dimension
        hidden_features = torch.nn.functional.relu(self.fc_hidden(resnet_features))
        
        return hidden_features  # Return the extracted features



