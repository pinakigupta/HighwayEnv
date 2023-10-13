from copy import deepcopy as dcp
import torch.nn as nn
from torch import multiprocessing
import os, shutil
os.environ["HDF5_USE_THREADING"] = "true"
import h5py
import copy
import sys
from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy
from torch.utils.data import Dataset, DataLoader
import torch
from generate_expert_data import extract_post_processed_expert_data  
import pygame  
import numpy as np
import seaborn as sns
from highway_env.utils import lmap
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import wandb
import json
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
import torchvision.models as models
from contextlib import ExitStack
from torchvision.models.video import R3D_18_Weights
from torchvision import transforms
import random
import math

from highway_env.envs.common.action import DiscreteMetaAction
ACTIONS_ALL = DiscreteMetaAction.ACTIONS_ALL
ACTIONS_LAT = DiscreteMetaAction.ACTIONS_LAT
ACTIONS_LONGI = DiscreteMetaAction.ACTIONS_LONGI
import gym
import numpy as np
from gym import spaces






def clear_and_makedirs(directory):
    # Clear the directory to remove existing files
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

class ZipfContextManager:
    def __init__(self, zip_filename, file_name):
        self.zip_filename = zip_filename
        self.file_name = file_name
        self.exit_stack = ExitStack()

    def __enter__(self):
        self.zipf = self.exit_stack.enter_context(zipfile.ZipFile(self.zip_filename, 'r'))
        self.file_in_zip = self.exit_stack.enter_context(self.zipf.open(self.file_name))
        self.hf = self.exit_stack.enter_context(h5py.File(self.file_in_zip, 'r', rdcc_nbytes=1024**3, rdcc_w0=0))
        return self.hf

    def __exit__(self, exc_type, exc_value, traceback):
        return self.exit_stack.__exit__(exc_type, exc_value, traceback)

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

class RandomPolicy(BasePolicy):
    def __init__(self, env, device, *args, **kwargs):
        super(RandomPolicy, self).__init__(
                                            *args, 
                                            observation_space=env.observation_space,
                                            action_space=env.action_space,
                                            **kwargs
                                          )
        # self.action_space = env.action_space
        # self.device = device
        self.n = self.action_space.n

    def _predict(self, obs, deterministic=False):
        # Generate random actions from a uniform distribution
        action = np.array(np.random.randint(0, self.n))
        return action, None
    
    def predict(self, obs):
        return self._predict(obs)
    
# Custom feature extractor for the action space
class ActionFeatureExtractor(nn.Module):
    def __init__(self, action_space):
        super(ActionFeatureExtractor, self).__init__()
        if isinstance(action_space, spaces.Discrete):
            self.embeddings = nn.Embedding(action_space.n, 32)
        elif isinstance(action_space, spaces.Box):
            self.embeddings = nn.Identity(action_space.shape[0])
    
    def forward(self, action):
        return self.embeddings(action)
    
def DefaultActorCriticPolicy(env, device, **policy_kwargs):
        state_dim = env.observation_space.high.shape[0]*env.observation_space.high.shape[1]
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
                                    observation_space=spaces.Dict({"obs": env.observation_space, "action": env.action_space}),
                                    action_space=env.action_space,
                                    lr_schedule=lr_schedule,
                                    **policy_kwargs
                                  )

        return policy   

class CustomDataset(Dataset):
    def __init__(self, data_file, device, pad_value=0, **kwargs):
        # Load your data from the file and prepare it here
        # self.data = ...  # Load your data into this variable
        self.keys_attributes = kwargs['keys_attributes'] 
        self.data_file = data_file
        self.data = extract_post_processed_expert_data(data_file, self.keys_attributes)
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
            for key in self.keys_attributes:
                if key in self.data:
                    try:
                        # Attempt to deserialize the data as JSON
                        value = json.loads(self.data[key][idx]) 
                    except:
                        # If JSON deserialization fails, assume it's a primitive type
                        value = self.data[key][idx]
                    value = torch.tensor(value, dtype=torch.float32)
                    # value = torch.tensor(self.data[key][idx], dtype=torch.float32)
                    sample[key] = value

        except Exception as e:
            pass
            # print(f' {e} , for , { id(self)}. key {key} , value {value} ')
            # raise e

        sample['obs'][:, -1] = 0

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

# Create a 3D ResNet model for feature extraction
class CustomVideoFeatureExtractor(BaseFeaturesExtractor):
    
    video_preprocessor = R3D_18_Weights.KINETICS400_V1.transforms()
    # Create a lambda function to add random Gaussian noise
    random_noise = lambda image: image + torch.randn(image.size()).to(image.device) * 0.05
    image_augmentations = [
                            # transforms.RandomResizedCrop(112),
                            transforms.RandomHorizontalFlip(),
                            # transforms.RandomGaussianNoise(std=(0, 0.05)),
                            random_noise,
                            transforms.ColorJitter(brightness=(0.7, 1.3), contrast=(0.7, 1.3))
                          ]

    def __init__(self, observation_space, hidden_dim=64, **kwargs):
        super(CustomVideoFeatureExtractor, self).__init__(observation_space, hidden_dim)

        if 'augment_image' in kwargs and kwargs['augment_image']:
            self.augment_image = True
        else:
            self.augment_image = False

        if not isinstance(observation_space, gym.spaces.Box):
            raise ValueError("Observation space must be continuous (e.g., image-based)")
        
        self.resnet = models.video.r3d_18(pretrained=True)  # You can choose a different ResNet variant
        # Remove the classification (fully connected) layer
        self.feature_extractor = nn.Sequential(*list(self.resnet.children())[:-1])
        self.set_grad_video_feature_extractor(requires_grad=False)

        self.height, self.width = observation_space.shape[1], observation_space.shape[2]

        # Additional fully connected layer for custom hidden feature dimension
        self.fc_hidden = nn.Sequential(
                                        nn.Linear(self.resnet.fc.in_features, hidden_dim),
                                        nn.Tanh(),
                                        nn.Linear(hidden_dim, hidden_dim),
                                        nn.Tanh(),
                                        nn.Linear(hidden_dim, hidden_dim)
                                    )

    def set_grad_video_feature_extractor(self, requires_grad=False):
        for param in self.feature_extractor.parameters():
            param.requires_grad = requires_grad

    def forward(self, observations):

        # Reshape observations to [batch_size, channels, stack_size, height, width]
        observations = observations.view(observations.size(0),  -1, 1, self.height, self.width)
        observations = torch.cat([observations, observations, observations], dim=2)
        # Normalize pixel values to the range [0, 1] (if necessary)
        # observations = observations / 255.0

        if self.augment_image:
            # Create a Compose transform to apply all of the transforms in the sequence to each frame in the video
            compose_transforms = transforms.Compose(self.image_augmentations)
            observations = compose_transforms(observations)
        
        observations = self.video_preprocessor(observations)

        # print(self.feature_extractor)
        # Pass input through the feature extractor (without the classification layer)
        resnet_features = self.feature_extractor(observations).squeeze()
        # Apply a fully connected layer for the custom hidden feature dimension
        hidden_features = torch.nn.functional.relu(self.fc_hidden(resnet_features))
        return hidden_features

def save_checkpoint(project, run_name, epoch, model, **kwargs):
    # if epoch is None:
    epoch = "final"
    model_path = f"models_archive/agent_{epoch}.pth"
    jit_model_path = f"models_archive/agent_{epoch}.pt"
    zip_filename = f"models_archive/agent.zip"
    torch.save( obj = model , f= model_path, pickle_protocol=5) 
    # torch.jit.trace(torch.load(model_path), torch.randn(1, *model.observation_space.shape)).save(jit_model_path)
    # os.remove(model_path)
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
        hdf5_train_file_names = [file_name for file_name in zipf.namelist() if file_name.endswith('.h5')  and kwargs['type'] in file_name]           
        for train_data_file in hdf5_train_file_names:
            if train_data_file not in visited_data_files:
                visited_data_files.add(train_data_file)
                with zipf.open(train_data_file) as file_in_zip:
                    print(f"Opening the data file {train_data_file}")
                    samples = CustomDataset(file_in_zip, device, keys_attributes = ['obs', 'act'])
                    print(f"Loaded custom data set for {train_data_file}")
                    new_key = 'acts'
                    old_key = 'act'
                    modified_dataset = []
                    for i in range(len(samples)):
                        my_dict = samples[i]
                        if old_key in my_dict and new_key not in my_dict:
                            my_dict[new_key] = my_dict.pop(old_key)
                        modified_dataset.append(my_dict)

                    # print('modified_dataset', modified_dataset)

                    train_datasets.append(modified_dataset)
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
    # all_actions = [sample['acts'] for sample in combined_train_dataset]
    # action_frequencies = np.bincount(all_actions)
    # class_weights = 1.0 / np.sqrt(action_frequencies)
    # # class_weights =  np.array([np.exp(-freq/action_frequencies.sum()) for freq in action_frequencies])
    # class_weights = class_weights / class_weights.sum()
    # print(" class_weights at the end ", class_weights, " action_frequencies ", action_frequencies)

    # # Calculate the least represented count
    # least_represented_count = np.min(action_frequencies)

    # # Get the number of unique action types
    # num_action_types = len(np.unique(all_actions))
    # num_samples=int(least_represented_count * num_action_types )
    # print(" class_weights ", class_weights, " num_samples ", num_samples, " original samples fraction ", num_samples/len(all_actions))
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

def process_batch(output_queue, input_queue, labels, worker_index,lock, batch_count, **kwargs):
    # local_policy_path = f'/tmp/validation_policy{worker_index}.pth'
    # shutil.copy('validation_policy.pth', local_policy_path)
    if lock.acquire(timeout=10):
        local_policy = torch.load('validation_policy.pth', map_location='cpu')
        lock.release()
    else:
        print(f"Couldn't fetch the validation policy for worker {worker_index}")
        return
    
    print(f"ID of val policy is {id(local_policy)}")
    output_samples =[]
    while True:
        if not input_queue.empty():
            batch = input_queue.get()
        else:
            time.sleep(0.1)
            continue
        # print(f"batch collected from worker {worker_index}")
        true_labels = batch['acts'].numpy()
        with torch.no_grad():
            # print(f"Lock acquired by {worker_index}")
            predicted_labels = [local_policy.predict(obs.numpy())[0] for obs in batch['obs']]
            _, log_prob, entropy = local_policy.evaluate_actions(batch['obs'], batch['acts'])
            if 'label_weights' in kwargs:
                true_labels         = true_labels @ kwargs['label_weights']
                predicted_labels    = predicted_labels @ kwargs['label_weights']
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels, average=None, labels=labels)
        recall = recall_score(true_labels, predicted_labels, average=None, labels=labels)
        f1 = f1_score(true_labels, predicted_labels, average=None, labels=labels)
        cross_entropy = -log_prob.mean().detach().cpu().numpy()
        entropy = entropy.mean().detach().cpu().numpy()
        conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=labels)
        output_sample = {
                            'accuracy': accuracy,
                            'precision': precision,
                            'recall': recall,
                            'f1': f1,
                            'cross_entropy': cross_entropy,
                            'entropy': entropy,
                            'conf_matrix':conf_matrix 
                        }
        output_queue.put(output_sample)
        # if lock.acquire(timeout=0.1):
        #     output_queue.put(output_samples)
        #     lock.release()
        # else:
        #     output_samples.append(output_sample)



def calculate_validation_metrics(val_data_loader, policy,zip_filename, **kwargs):
    true_labels = []
    predicted_labels = []
    # Iterate through the validation data and make predictions
    # CustomDataLoader(zip_filename, device, visited_data_files, batch_size, n_cpu, type='train')
    print(f"Calcuating validaton metrices for {kwargs['type']}-------------> ")
    # val_data_loader = CustomDataLoader(
    #                                     zip_filename, 
    #                                     **{**kwargs,
    #                                         'type':kwargs['type'],
    #                                         'validation': True,
    #                                         'visited_data_files': []
    #                                       }
    #                                   ) 
    
    conf_matrix = None
    accuracies = []
    precisions = []
    recalls = []
    cross_entropies = []
    entropies = []
    f1s = []
    labels = list(range(len(ACTIONS_LAT.keys())*len(ACTIONS_LAT.keys())))
    policy.eval()
    batch_count = 0

    manager = multiprocessing.Manager()
    lock = manager.Lock()
    torch.save(policy,'validation_policy.pth')
    # Create a list to store process objects
    processes = []
    output_queue = manager.Queue()
    input_queue = manager.Queue()
    num_workers =   max(kwargs['n_cpu'],1)

    for i in range(num_workers):
        
        worker_process = multiprocessing.Process(
                                                    target=process_batch, 
                                                    args=(
                                                            output_queue, 
                                                            input_queue, 
                                                            labels, 
                                                            i,
                                                            lock,
                                                            batch_count
                                                        ),
                                                    kwargs = kwargs, 
                                                )
        processes.append(worker_process)
        worker_process.start()

    progress_bar = tqdm(total=kwargs['val_batch_count'], desc= f'{kwargs["type"]} progress')
    conf_matrix = np.array([])

    def calculate_metrics():
        nonlocal accuracies, precisions, recalls, f1s, cross_entropies, entropies, conf_matrix
        result = output_queue.get()
        accuracies.append(result['accuracy'])
        precisions.append(result['precision'])
        recalls.append(result['recall'])
        f1s.append(result['f1'])
        cross_entropies.append(result['cross_entropy'])
        entropies.append(result['entropy'])
        if conf_matrix.size == 0 :
            conf_matrix = result['conf_matrix']
        else:
            conf_matrix += result['conf_matrix']
        progress_bar.update(1)
            
    for batch in val_data_loader:
        sample = {key: value.to(torch.device('cpu')) for key, value in batch.items()}
        try:
            input_queue.put(sample)
            if input_queue.qsize() > 1.25*kwargs['val_batch_count']:
                break
        except Exception as e:
            print(e)
        calculate_metrics()

    while len(accuracies) < kwargs['val_batch_count']:
        if output_queue.empty():
            time.sleep(0.1)
            return
        calculate_metrics()


    progress_bar.close()



    # Wait for all processes to finish
    for process in processes:
        print(f'terminating process {process.pid}')
        process.terminate()

                

    # Calculate evaluation metrics
    axis = 0
    accuracy  = np.mean(accuracies,  axis=axis) 
    precision = np.mean(precisions, axis=axis)
    recall    = np.mean(recalls, axis=axis)
    f1        = np.mean(f1s, axis=axis)
    cross_entropies = np.mean(cross_entropies)
    entropies = np.mean(entropies)

    # Print the metrics
    print("Accuracy:", accuracy, np.mean(accuracy))
    print("cross_entropy:", np.mean(cross_entropies), "entropy:", np.mean(entropies))
    print("Precision:", precision, np.mean(precision))
    print("Recall:", recall, np.mean(recall))
    print("F1 Score:", f1, np.mean(f1))



    if True:
        plt.figure(figsize=(8, 6))
        class_labels = labels
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title(f"Confusion Matrix for {kwargs['type']}")
        heatmap_png = f'heatmap_{kwargs["type"]}.png'
        plt.savefig(heatmap_png)

    if False: # For now keep this local as the remote artifact size is growing too much
        with zipfile.ZipFile(zip_filename, 'a') as zipf:
            zipf.write(heatmap_png, arcname=training_kwargs['plot_path'])
    # plt.show()  
    # print("saved confusion matrix")

    while any(p.is_alive() for p in processes):
        alive_workers = [worker for worker in processes if worker.is_alive()]
        for worker_process in alive_workers:
            os.kill(worker_process.pid, signal.SIGKILL)
        time.sleep(0.25)  # Sleep for a second (you can adjust the sleep duration)
        print([worker.pid for worker in processes if worker.is_alive()])

    metrics = {
               'accuracy':      accuracy,
               'precision':     np.mean(precision), 
               'recall':        np.mean(recall), 
               'f1':            np.mean(f1),
               'cross_entropy': np.mean(cross_entropies),
               'entropy':       np.mean(entropies),

            }
    return metrics


class CustomImageExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, hidden_dim=64):
        super(CustomImageExtractor, self).__init__(observation_space, hidden_dim)
        self.channels = 3
        if not isinstance(observation_space, gym.spaces.Box):
            raise ValueError("Observation space must be continuous (e.g., image-based)")
        self.height, self.width = observation_space.shape[1], observation_space.shape[2]
        
        # Define a pretrained ResNet model as the feature extractor trunk
        self.resnet = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)
        # self.resnet.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Adjust the last fully connected layer for feature dimension control
        # self.resnet.fc = nn.Linear(self.resnet.fc.in_features, hidden_dim)
        self.feature_extractor = nn.Sequential(*list(self.resnet.children())[:-1])

        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        self.lstm = nn.LSTM(input_size=self.resnet.fc.in_features, hidden_size=hidden_dim, num_layers=4, batch_first=True)
        
        # Additional fully connected layer for custom hidden feature dimension
        self.fc_hidden = nn.Sequential(
                                        nn.Linear(self.lstm.hidden_size, hidden_dim),
                                        nn.Tanh(),
                                        nn.Linear(hidden_dim, hidden_dim),
                                        nn.Tanh(),
                                        nn.Linear(hidden_dim, hidden_dim)
                                      )
    
    def forward(self, observations):

        # Reshape observations to [batch_size, stack_size, height, width]
        observations = observations.view(observations.size(0), -1, 1, self.height, self.width)
        observations = observations / 255.0

        # Expand the grayscale image along the channel dimension to create an RGB image
        observations = observations.repeat(1, 1, self.channels , 1, 1)        # Normalize pixel values to the range [0, 1] (if necessary)
        
        # Reshape the input for feature extraction
        batch_size, seq_len, c, h, w = observations.size()
        observations = observations.view(batch_size * seq_len, c, h, w)

        # Forward pass through the ResNet trunk for feature extraction
        resnet_features = self.feature_extractor(observations).squeeze()

        # Flatten the features and pass through the LSTM
        resnet_features = resnet_features.view(batch_size, seq_len, -1)  # Flatten the spatial dimensions

          # Pass through the LSTM layer
        lstm_features, _ = self.lstm(resnet_features)

        # Reshape the output of the LSTM to match the expected input size of fc_hidden
        lstm_features = lstm_features.contiguous().view(batch_size , seq_len, -1)
        lstm_features = lstm_features[:, -1, :]  # Select the last time step

        # Apply a fully connected layer for the custom hidden feature dimension
        hidden_features = torch.nn.functional.relu(self.fc_hidden(lstm_features))
        
        return hidden_features  # Return the extracted features


class CustomDataLoader: # Created to deal with very large data files, and limited memory space
    def __init__(self, zip_filename, device, visited_data_files, batch_size, n_cpu, validation=False, verbose=False, **kwargs):
        self.kwargs = kwargs
        self.zip_filename = zip_filename
        self.device = device
        self.visited_data_files = visited_data_files
        self.batch_size = batch_size
        self.n_cpu =  n_cpu
        self.verbose = verbose
        self.is_validation = validation or self.kwargs['type'] == 'val'
        with zipfile.ZipFile(self.zip_filename, 'r') as zipf:
            self.hdf5_train_file_names = [file_name for file_name in zipf.namelist() if file_name.endswith('.h5') and kwargs['type'] in file_name]
        print('self.hdf5_file_names ', self.hdf5_train_file_names)
        if 'chunk_size' in kwargs:
            self.chunk_size = kwargs['chunk_size']
        else:
            self.chunk_size = 7500
        self.all_worker_indices = self.calculate_worker_indices(self.chunk_size)
        self.manager = multiprocessing.Manager()
        self.all_obs = self.manager.list()
        self.all_acts = self.manager.list()
        self.all_dones = self.manager.list()
        self.all_kin_obs = self.manager.list()
        self._reset_tuples()
        self.batch_no = 0
        # self.pool = multiprocessing.Pool()
        self.total_samples = self.manager.dict()
        self.total_chunks = self.manager.dict()
        self.lock = self.manager.Lock()
        self.iter = 0
        self.step_num = 0
        self.keys_to_consider = [
                                    'obs', 
                                    # 'kin_obs', 
                                    'acts', 
                                    'dones'
                                    ]
        self.epoch = 0


    def _reset_tuples(self):
        self.all_obs = []
        self.all_kin_obs = []
        self.all_acts = []
        self.all_dones = []
                   
    def launch_reader_workers(self, reader_queue):
        # Create and start worker processes
        
        processes = []
        all_chunks = [(file_name, chunk) for file_name, chunks in self.total_chunks.items() for chunk in chunks]
        random_samples = random.sample(all_chunks, min(self.n_cpu, len(all_chunks)))
        for i, (file_name, chunk_index) in enumerate(random_samples):
            process = multiprocessing.Process(
                                                target=self.reader_worker, 
                                                args=(
                                                        i, 
                                                        self.total_chunks, 
                                                        self.total_samples,
                                                        file_name,
                                                        chunk_index,
                                                        reader_queue
                                                     )
                                             )
            process.start()
            processes.append(process)
        return processes
    
    def launch_writer_workers(self, samples_queue, total_samples_for_chunk):
        # Create and start worker processes
        
        processes = []
        for i in range(self.n_cpu):
            process = multiprocessing.Process(
                                                target=self.writer_worker, 
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
    
    @staticmethod
    def destroy_workers(processes):
        for process in processes:
            process.terminate()

        while any(p.is_alive() for p in processes):
            alive_workers = [worker for worker in processes if worker.is_alive()]
            for worker_process in alive_workers:
                os.kill(worker_process.pid, signal.SIGKILL)
            time.sleep(0.1)  # Sleep for a second (you can adjust the sleep duration)
            # print([worker.pid for worker in processes if worker.is_alive()])
            
    def load_batch(self, batch_samples):

        batch = {}

        try:
            for key in self.keys_to_consider:
                values = [list(sample[key].values()) if isinstance(sample[key], dict) else sample[key] for sample in batch_samples] 
                if key == 'acts' and 'label_weights' in self.kwargs:
                    values = values @ self.kwargs['label_weights']
                batch[key] = torch.tensor(values, dtype=torch.float32).to(self.device)
        except Exception as e:
            print(batch_samples[0].keys())
            for k, v,in batch_samples.items():
                print(f'{k} length is {len(v)}')
            raise e
        

        # Clear GPU memory for individual tensors after creating the batch
        for sample in batch_samples:
            for key in self.keys_to_consider:
                if key in sample:
                    del sample[key]
                    
        return batch   
     
    def __iter__(self):
        num_yielded_batches = 0 
        while True: # Keep iterating till num batches is met
            if self.verbose:
                print(f"++++++++++++++++++ ITER PASS {self.iter} +++++++++++++++++++")
            self.iter += 1
            for file_name in self.hdf5_train_file_names:
                with ZipfContextManager(self.zip_filename, file_name) as hf:
                    self.total_samples[file_name] = len(hf['act'])
                    self.total_chunks[file_name] = list(range(math.ceil(self.total_samples[file_name]/self.chunk_size)))
            for batch in self.iter_once_all_files(): # Keep iterating
                yield batch
                num_yielded_batches += 1
                
                if self.is_validation and num_yielded_batches >= self.kwargs['val_batch_count']:
                    return  # Exit the generator loop
            self.epoch += 1
            print(f"Data loader Epoch count {self.epoch}")
            if self.is_validation: # Only one iteration through all the files is file
                break

    def reader_worker(
                        self, 
                        worker_id, 
                        total_chunks, 
                        total_samples,
                        file_name,
                        chunk_num,
                        reader_queue
                     ):
        # file_name = random.sample(total_samples.keys(), 1)[0]
        # Flatten the list of chunks along with their corresponding file names
        
        with ZipfContextManager(self.zip_filename, file_name) as hf:
            # if total_chunks[file_name] and total_samples[file_name]:
            # print(f"Acquired lock from worker {worker_id} for file_name {file_name}. Total chunks {total_chunks[file_name]} and total samples {total_samples[file_name]}")
            # chunk_num = random.sample(total_chunks[file_name], 1)[0]
            temp = total_chunks[file_name]
            temp.remove(chunk_num)
            total_chunks[file_name] = temp

            all_indices = list(range(total_samples[file_name]))
            starting_sample = self.chunk_size*chunk_num
            total_samples_for_chunk = min(total_samples[file_name]-starting_sample, self.chunk_size)
            chunk_indices = all_indices[starting_sample:starting_sample + total_samples_for_chunk]
            chunk = {}
            for key in [
                                    'obs', 
                                    # 'kin_obs', 
                                    'act', 
                                    'dones'
                                    ]:
                    try:
                        # Attempt to deserialize the data as JSON
                        chunk[key] = [json.loads(data_str) for data_str in hf[key][chunk_indices]]
                    except:
                        # If JSON deserialization fails, assume it's a primitive type
                        chunk[key] = hf[key][chunk_indices]
                                        

            reader_queue.put(chunk)
            # print(f"Acquired lock from worker {worker_id} . Accumulated samples of size {len(chunk['acts'])} for file_name {file_name}")


                
    # MAX_CHUNKS = 5
    def iter_once_all_files(self):
        self.step_num = 0 
        while self.total_samples:
            # print(f"Launching all reader processes for step {self.step_num}", flush=True)
            self.all_acts = []
            self.all_obs = []
            self.all_kin_obs = []
            self.all_dones = []
            reader_queue = self.manager.Queue()
            reader_processes = self.launch_reader_workers(reader_queue)
            # print(f"Launched all reader processes for step {self.step_num}", flush=True)



            for process in reader_processes:
                process.join()

            while(not reader_queue.empty()):
                sample = reader_queue.get()
                if 'obs' in sample:
                    self.all_obs.extend(sample['obs'])
                if 'kin_obs' in sample:
                    self.all_kin_obs.extend(sample['kin_obs'])
                if 'act' in sample:
                    self.all_acts.extend(sample['act'])
                if 'dones' in sample:
                    self.all_dones.extend(sample['dones'])
                time.sleep(0.01)
                # print('length', len(all_acts))



            for file_name in self.hdf5_train_file_names:
                if not self.total_chunks[file_name] and file_name in self.total_samples:
                    del self.total_samples[file_name] # No more chunk left in this file

            self.destroy_workers(reader_processes)

            # print(f"Joined all reader processes for step {self.step_num}. length', {len(self.all_acts)}")

            # Randomize the aggregated chunks across all files by shuffling
            all_indices = list(range(len(self.all_acts) ))
            random.shuffle(all_indices)
            self.all_obs[:] = [self.all_obs[i] for i in all_indices if self.all_obs]
            self.all_kin_obs[:] = [self.all_kin_obs[i] for i in all_indices if self.all_kin_obs] 
            self.all_acts[:] = [self.all_acts[i] for i in all_indices if self.all_acts]
            self.all_dones[:] = [self.all_dones[i] for i in all_indices if self.all_dones]

            # print(f"Collected all samples of size {len(self.all_obs)}")    
            #Yield chunks as they come
            for batch in self.generate_batches_from_one_chunk(len(self.all_acts) ,self.step_num):
                yield batch



            self.step_num +=1


    def generate_batches_from_one_chunk(self, total_samples_for_chunk ,step_num):
        if self.verbose:
            print(f"Launching writer worker processes for step {step_num}", flush=True)
        samples_queue = self.manager.Queue(maxsize=self.batch_size * 2)  # Multiprocessing Queue for accumulating samples
        self.all_worker_indices = self.calculate_worker_indices(total_samples_for_chunk)
        worker_processes = self.launch_writer_workers(samples_queue, total_samples_for_chunk)

        all_samples = [] 
        progress_bar = tqdm(total=total_samples_for_chunk, desc=f'Processing writer chunk {step_num}')
        total_samples_yielded = 0
        while total_samples_yielded < 0.9*total_samples_for_chunk:
            sample = samples_queue.get()
            if sample is None:
                continue
            all_samples.append(sample)
            # print("total_samples_yielded ",total_samples_yielded, " total_samples_for_chunk ", total_samples_for_chunk,\
            #        " all_samples ", len(all_samples))
            if len(all_samples)-total_samples_yielded >= self.batch_size:
                yield self.load_batch(all_samples[total_samples_yielded:total_samples_yielded+self.batch_size])
                total_samples_yielded += self.batch_size
                # del all_samples[:self.batch_size]
                progress_bar.update(self.batch_size)
                # time.sleep(0.01)
        progress_bar.close()
        # self._reset_tuples()
        # Collect and yield batches from each process
        self.destroy_workers(worker_processes)
        if self.verbose:
            print(f" Terminated all writer process for  step {step_num}", flush=True)




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

    
    def writer_worker(self, worker_id, total_samples , worker_indices, samples_queue):
                    
        self.count = 0
        worker_indices = [index for index in worker_indices if index < total_samples]
        worker_indices.sort()
        for index in worker_indices:
            self.count +=1
            # Append the sample to the list
            try:
                sample = {}
                if self.all_obs:
                    sample['obs'] = self.all_obs[index]
                if self.all_kin_obs:
                    sample['kin_obs'] = self.all_kin_obs[index]
                if self.all_acts:
                    sample['acts'] = self.all_acts[index]
                if self.all_dones:
                    sample['dones'] = self.all_dones[index]
                if sample:
                    sample['obs'][:, -1 ] = 0
                    samples_queue.put(sample)
            except (IndexError, EOFError, BrokenPipeError) as e:
                # print(f'Error {e} accessing index {index} in writer worker {worker_id}. Length of obs {len(self.all_obs)}')
                pass
            except Exception as e:
                pass
            # time.sleep(0.01)
                        
        # samples_queue.put(None)

