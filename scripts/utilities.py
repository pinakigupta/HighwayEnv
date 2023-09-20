from copy import deepcopy as dcp
import torch.nn as nn
from torch import multiprocessing
import os, shutil
os.environ["HDF5_USE_THREADING"] = "true"
import h5py
import copy
# import h5pyd as h5py
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
from attention_network import EgoAttentionNetwork
import gymnasium as gym
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
import zipfile
import time
import signal
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler, Subset

from highway_env.envs.common.action import DiscreteMetaAction
ACTIONS_ALL = DiscreteMetaAction.ACTIONS_ALL

def process_file(train_data_file, result_queue, device, lock):
    processed_data = CustomDataset(train_data_file, device)
    with lock:
        result_queue.put([processed_data])

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
        self.data = extract_post_processed_expert_data(data_file)
        self.pad_value = pad_value
        self.device = device
        # print(" data lengths ", len(self.exp_obs), len(self.exp_acts), len(self.exp_dones))
        return

    def __len__(self):
        # print("data length for custom data set ",id(self), " is ", len(self.exp_acts),  len(self.exp_obs))
        return len(self.data['acts'])

    def __getitem__(self, idx):
        sample = {}
        try:
            if 'obs' in self.data:
                observation = torch.tensor(self.data['obs'][idx], dtype=torch.float32)
                sample['obs'] = observation
            if 'kin_obs' in self.data:
                kin_observation = torch.tensor(self.data['kin_obs'][idx], dtype=torch.float32)
                sample['obs'] = kin_observation
            if 'acts' in self.data:
                action = torch.tensor(self.data['acts'][idx], dtype=torch.float32)
                sample['acts'] = action
            if 'dones' in self.data:
                dones = torch.tensor(self.data['dones'][idx], dtype=torch.float32)
                sample['dones'] = dones
        except Exception as e:
            print(e , "for ", id(self), len(self.exp_obs), len(self.exp_acts), len(self.exp_dones))
            raise e

        return sample
        


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
    # Extract the names of the HDF5 files from the zip archive
    with zipfile.ZipFile(zip_filename, 'r') as zipf:
        print(" File handle for the zip file opened ")
        hdf5_train_file_names = [file_name for file_name in zipf.namelist() if file_name.endswith('.h5') and "train" in file_name]           
        for train_data_file in hdf5_train_file_names:
            if train_data_file not in visited_data_files:
                visited_data_files.add(train_data_file)
                with zipf.open(train_data_file) as file_in_zip:
                    print(f"Opening the data file {train_data_file}")
                    train_datasets.extend([CustomDataset(file_in_zip, device)])
                    print(f"Dataset appended for  {train_data_file}")

    print("DATA loader scanned all files")

    # shutil.rmtree(extract_path)

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
                    # in_memory_file = io.BytesIO(file_in_zip.read())
                    data = extract_post_processed_expert_data(file_in_zip)
                    # in_memory_file.close()
                    predicted_labels.extend([policy.predict(obs)[0] for obs in data['obs']])
                    true_labels.extend(data['acts'])
                    

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

import random 
class CustomDataLoader: # Created to deal with very large data files, and limited memory space
    def __init__(self, zip_filename, device, visited_data_files, batch_size, n_cpu):
        self.zip_filename = zip_filename
        self.device = device
        self.visited_data_files = visited_data_files
        self.batch_size = batch_size
        self.n_cpu =  n_cpu 
        with zipfile.ZipFile(self.zip_filename, 'r') as zipf:
            self.hdf5_train_file_names = [file_name for file_name in zipf.namelist() if file_name.endswith('.h5') and "train" in file_name]
        self.chunk_size = 10000  # Assume a large number of samples
        self.all_worker_indices = self.calculate_worker_indices(self.chunk_size)
        self.manager = multiprocessing.Manager()
        self.all_obs = self.manager.list()
        self.all_acts = self.manager.list()
        self.all_dones = self.manager.list()
        self.all_kin_obs = self.manager.list()
        self._reset_tuples()
        self.batch_no = 0
        self.pool = multiprocessing.Pool()

    def _reset_tuples(self):
        self.all_obs = []
        self.all_kin_obs = []
        self.all_acts = []
        self.all_dones = []
                   

    def launch_workers(self, samples_queue, total_samples_for_chunk):
        # Create and start worker processes
        
        processes = []
        for i in range(self.n_cpu):
            process = multiprocessing.Process(
                                                target=self.worker, 
                                                args=(
                                                        i, 
                                                        total_samples_for_chunk,
                                                        self.all_worker_indices[i], 
                                                        samples_queue
                                                     )
                                             )
            process.start()
            processes.append(process)
        return processes
    
    def destroy_workers(self, processes):
        for process in processes:
            process.terminate()

        while any(p.is_alive() for p in processes):
            alive_workers = [worker for worker in processes if worker.is_alive()]
            for worker_process in alive_workers:
                os.kill(worker_process.pid, signal.SIGKILL)
            time.sleep(0.1)  # Sleep for a second (you can adjust the sleep duration)
            # print([worker.pid for worker in processes if worker.is_alive()])
        

    
    def load_batch(self, batch_samples):
        batch = {
            'obs_': torch.tensor([sample['obs'] for sample in batch_samples], dtype=torch.float32).to(self.device),
            'obs': torch.tensor([sample['kin_obs'] for sample in batch_samples], dtype=torch.float32).to(self.device),
            'acts': torch.tensor([sample['acts'] for sample in batch_samples], dtype=torch.float32).to(self.device),
            'dones': torch.tensor([sample['dones'] for sample in batch_samples], dtype=torch.float32).to(self.device),
            # Add other keys as needed
        }

        # Clear GPU memory for individual tensors after creating the batch
        for sample in batch_samples:
            del sample['obs']
            del sample['kin_obs']
            del sample['acts']
            del sample['dones']
        return batch   
     

    def __iter__(self):
        while True:
            for batch in self.iter_once_all_files(): # Keep iterating
                yield batch

    def __iter_once__(self):    
        for train_data_file in self.hdf5_train_file_names :
            # if train_data_file  in self.visited_data_files:
            #     continue
            # self.visited_data_files.add(train_data_file)
            for item in self.iter_once_one_file(train_data_file):
                yield item
            # print(f"Iter once through file {train_data_file}")


    def iter_once_all_files(self):
        # print(f"Inside __iter_once_this_file__ for file {train_data_file}")
        # Loop through each file name and open the files
        # file_handles = []
        total_samples = {}
        for file_name in self.hdf5_train_file_names:
            total_samples[file_name] = None

        chunk_num = 0
        while total_samples: # chunk iteration. Will read till any file has any chunk left.
            # scanned_samples = 0
            for file_name in self.hdf5_train_file_names: # File reading is still not complete
                if file_name not in total_samples:
                    continue
                with zipfile.ZipFile(self.zip_filename, 'r') as zipf:
                    with zipf.open(file_name) as file_in_zip:
                        with h5py.File(file_in_zip, 'r', rdcc_nbytes=1024**3, rdcc_w0=0) as hf:
                            if total_samples[file_name] is None:
                                total_samples[file_name] = len(hf['act'])
                        # for file_name, hf in file_handles:
                            # print(f"Worker {worker_id}: Read the data file in place {hdf5_file_to_parse}")
                            print(f"Opening handle for for file {file_name} for chunk {chunk_num}")
                            all_indices = list(range(total_samples[file_name]))
                            # random.shuffle(all_indices)
                            
                            scanned_samples = self.chunk_size*chunk_num
                            if scanned_samples <= total_samples[file_name]:
                                total_samples_for_chunk = min(total_samples[file_name]-scanned_samples, self.chunk_size)
                                chunk_indices = all_indices[scanned_samples:scanned_samples + total_samples_for_chunk]
                                # chunk_indices.sort()
                                # shuffled_indices = list(range(len(chunk_indices)))
                                # random.shuffle(shuffled_indices)
                                self.all_obs.extend(hf['obs'][chunk_indices])
                                self.all_kin_obs.extend(hf['kin_obs'][chunk_indices])
                                self.all_acts.extend(hf['act'][chunk_indices])
                                self.all_dones.extend(hf['dones'][chunk_indices])
                                scanned_samples += self.chunk_size
                                print(f" scanned_samples {scanned_samples } for file {file_name}", flush=True)
                                # self.deploy_workers(file_name, chunk_num)
                            else:
                                del total_samples[file_name] # No more chunk left in this file

            # Randomize the aggregated chunks across all files
            all_indices = list(range(len(self.all_acts) ))
            random.shuffle(all_indices)
            self.all_obs[:] = [self.all_obs[i] for i in all_indices]
            self.all_kin_obs[:] = [self.all_kin_obs[i] for i in all_indices]
            self.all_acts[:] = [self.all_acts[i] for i in all_indices]
            self.all_dones[:] = [self.all_dones[i] for i in all_indices]

            #Yield after 1 chunk has been read across all the files (for the files that had a readable chunk left)
            for batch in self.generate_batches_from_one_chunk(len(self.all_acts) ,chunk_num):
                yield batch
            chunk_num +=1
            self.all_acts = []
            self.all_obs = []
            self.all_kin_obs = []
            self.all_dones = []

    def generate_batches_from_one_chunk(self, total_samples_for_chunk ,chunk_num):
        print(f"Launching worker processes for chunk {chunk_num}", flush=True)
        samples_queue = self.manager.Queue(maxsize=self.batch_size * 2)  # Multiprocessing Queue for accumulating samples
        self.all_worker_indices = self.calculate_worker_indices(total_samples_for_chunk)
        processes = self.launch_workers(samples_queue, total_samples_for_chunk)

        all_samples = [] 
        progress_bar = tqdm(total=total_samples_for_chunk, desc=f'Processing  chunk {chunk_num}')
        total_samples_yielded = 0
        while total_samples_yielded < 0.9*total_samples_for_chunk:
            sample = samples_queue.get()
            if sample is None:
                continue
            all_samples.append(sample)
            # print("total_samples_yielded ",total_samples_yielded)
            if len(all_samples)-total_samples_yielded >= self.batch_size:
                yield self.load_batch(all_samples[total_samples_yielded:total_samples_yielded+self.batch_size])
                total_samples_yielded += self.batch_size
                # del all_samples[:self.batch_size]
            progress_bar.update(1)
        progress_bar.close()
        # self._reset_tuples()
        # Collect and yield batches from each process
        self.destroy_workers(processes)
        print(f" Terminated all process for  chunk {chunk_num}", flush=True)




    # def deploy_workers(self, train_data_file, chunk_num):


    def calculate_worker_indices(self, total_samples):
        # Calculate indices for each worker
        indices = list(range(total_samples))
        random.shuffle(indices)  # Shuffle the indices randomly
        chunk_size = total_samples // self.n_cpu

        worker_indices = []
        for i in range(self.n_cpu):
            start = i * chunk_size
            end = start + chunk_size if i < self.n_cpu - 1 else total_samples
            worker_indices.append(indices[start:end])

        return worker_indices

    
    def worker(self, worker_id, total_samples , worker_indices, samples_queue):
                    
        self.count = 0
        worker_indices = [index for index in worker_indices if index < total_samples]
        worker_indices.sort()
        for index in worker_indices:
            self.count +=1
            # Append the sample to the list
            try:
                samples_queue.put(
                                        {
                                            "obs": self.all_obs[index], 
                                            "kin_obs": self.all_kin_obs[index],
                                            "acts": self.all_acts[index], 
                                            "dones": self.all_dones[index]
                                        }
                                    )
            except IndexError:
                pass
            # time.sleep(0.01)
                        
        # samples_queue.put(None)

