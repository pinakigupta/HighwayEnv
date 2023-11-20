from copy import deepcopy as dcp
import torch.nn as nn
from torch import multiprocessing
import os, shutil
from contextlib import redirect_stdout
import io
os.environ["HDF5_USE_THREADING"] = "true"
import h5py
import sys
from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy
from torch.utils.data import Dataset, DataLoader
import torch
from generate_expert_data import extract_post_processed_expert_data, retrieve_agent
import pygame  
import numpy as np
import seaborn as sns
from highway_env.utils import lmap
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import wandb
import json
from tqdm import tqdm
from attention_network import EgoAttentionNetwork
import gymnasium as gyms
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
import functools
from gymnasium.spaces import Space as Space

from highway_env.envs.common.action import DiscreteMetaAction
ACTIONS_ALL = DiscreteMetaAction.ACTIONS_ALL
ACTIONS_LAT = DiscreteMetaAction.ACTIONS_LAT
ACTIONS_LONGI = DiscreteMetaAction.ACTIONS_LONGI
import gym
import numpy as np
from gym import spaces
import tracemalloc

def clear_and_makedirs(directory):
    # Clear the directory to remove existing files
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)





def print_stack_size():
    while True:
        current_size, peak_size = tracemalloc.get_traced_memory()
        print(f"Current memory size: {current_size / (1024 * 1024):.2f} MB")
        time.sleep(15)  # Adjust the interval as needed

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
    
                
def DefaultActorCriticPolicy(env, device, **policy_kwargs):
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
                                    # observation_space= spaces.Dict({"obs": env.observation_space, "action": env.action_space}),
                                    observation_space = env.observation_space,
                                    action_space=env.action_space,
                                    lr_schedule=lr_schedule,
                                    share_features_extractor=True,
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
        prev_sample = {}
        try:
            for key in self.keys_attributes:
                if key in self.data:
                    try:
                        # Attempt to deserialize the data as JSON
                        value = json.loads(self.data[key][idx])
                        prev_value =  json.loads(self.data[key][idx-1]) if idx>1 else value
                    except:
                        # If JSON deserialization fails, assume it's a primitive type
                        value = self.data[key][idx]
                        prev_value = self.data[key][idx-1] if idx>1 else value
                    sample[key] = torch.tensor(value, dtype=torch.float32).to(self.device)
                    prev_sample[key] = torch.tensor(prev_value, dtype=torch.float32).to(self.device)

        except Exception as e:
            print(f' {e} , for , { id(self)}. key {key} , value {value} ')
            pass
            # raise e

        sample['obs'][:, -1].fill_(0)
        sample['obs'] = torch.cat([ sample['obs'].view(-1), prev_sample['act'].view(1)], dim=0)
        # sample['obs'] = {'obs':sample['obs'], 'action':prev_sample['act']}

        return sample 
        
        
def display_vehicles_attention(agent_surface, sim_surface, env, extractor, device,  min_attention=0.01):
        v_attention = compute_vehicles_attention(env, extractor, device)
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
        if not v_attention:
            return
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

def compute_vehicles_attention(env,extractor, device):
    observer = env.env.observation_type
    obs = observer.observe()
    obs_t = torch.tensor(obs[None, ...], dtype=torch.float).to(device)
    attention = extractor.get_attention_matrix(obs_t)
    attention = attention.squeeze(0).squeeze(1).detach().cpu().numpy()
    ego, others, mask = extractor.split_input(obs_t)
    mask = mask.squeeze()
    v_attention = {}
    obs_type = env.observation_type
    if hasattr(obs_type, "agents_observation_types"):  # Handle multi-model observation
        obs_type = obs_type.agents_observation_types[0]
    close_vehicles =       env.road.close_vehicles_to(env.vehicle,
                                                        env.PERCEPTION_DISTANCE,
                                                        count=observer.vehicles_count - 1,
                                                        see_behind=observer.see_behind,
                                                        sort=observer.order == "sorted")        
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
        if close_vehicles:
            vehicle = min(close_vehicles, key=lambda v: np.linalg.norm(v.position - v_position))
            v_attention[vehicle] = attention[:, v_index]
    return v_attention

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

def process_zip_file(result_queue, train_data_file, visited_data_files, zip_filename, device, lock):
    visited_filepath = zip_filename + train_data_file
    new_key = 'acts'
    old_key = 'act'
    modified_dataset = []

    # Acquire the manager lock before checking and updating the set
    with lock:
        if visited_filepath in visited_data_files:
            # The file has already been processed by another worker, skip it
            return modified_dataset
        visited_data_files.add(visited_filepath)

    with zipfile.ZipFile(zip_filename, 'r') as zipf, zipf.open(train_data_file) as file_in_zip:
        print(f"Processing the data file {train_data_file}")
        
        samples = CustomDataset(file_in_zip, device, keys_attributes=['obs', 'act'])
        for i in range(len(samples)):
            my_dict = samples[i]
            if old_key in my_dict and new_key not in my_dict:
                my_dict[new_key] = my_dict.pop(old_key)
            modified_dataset.append(my_dict)
        # return modified_dataset
    result_queue.put(modified_dataset)

        
def create_dataloaders(zip_filename, train_datasets, device, visited_data_files, **kwargs):
    # Extract the names of the HDF5 files from the zip archive
    with zipfile.ZipFile(zip_filename, 'r') as zipf:
        hdf5_train_file_names = [file_name for file_name in zipf.namelist() if file_name.endswith('.h5') and kwargs['type'] in file_name]

    # Create a pool of worker processes
    num_workers = 10
    with redirect_stdout(io.StringIO()): 
            # Create a manager to create a lock that can be shared among processes
        with multiprocessing.Manager() as manager:
        # Create a lock using the manager
            lock = manager.Lock()
            result_queue = manager.Queue()
            processes = []

            for train_data_file in hdf5_train_file_names:
                # Create a new process for each file
                p = multiprocessing.Process(target=process_zip_file, args=(result_queue, train_data_file, visited_data_files, zip_filename, device, lock))
                p.start()
                processes.append(p)

            # Wait for all processes to finish
            for p in processes:
                p.join()

            # Retrieve results from the queue
            while not result_queue.empty():
                modified_dataset = result_queue.get()
                if modified_dataset:
                    train_datasets.extend(modified_dataset)


    # Combine the results into the train_datasets list
    # train_datasets.extend(modified_datasets)
    print("All datasets appended.")

    # shutil.rmtree(extract_path)

    # val_data_loader                                             =  CustomDataLoader(
    #                                                                             zip_filename, 
    #                                                                             device=device,
    #                                                                             batch_size=kwargs['batch_size'],
    #                                                                             n_cpu=kwargs['n_cpu'],
    #                                                                             val_batch_count=np.inf,
    #                                                                             chunk_size=500,
    #                                                                             type= kwargs['type'],
    #                                                                             plot_path=None,
    #                                                                             visited_data_files = visited_data_files
    #                                                                         ) 
    # for batch in val_data_loader:
    #     train_datasets.append(batch)


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
                                        shuffle=True,
                                        # sampler=sampler,
                                        drop_last=True,
                                        num_workers=kwargs['n_cpu'],
                                        # pin_memory=True,
                                        # pin_memory_device=device,
                                 ) 
    return train_data_loader

def process_validation_batch(output_queue, input_queue, labels, worker_index,lock, batch_count=None, **kwargs):
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
                                                    target=process_validation_batch, 
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
            if len(accuracies) > kwargs['val_batch_count']:
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
        plt.title(f"Confusion Matrix for {kwargs['type']} for {zip_filename}")
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
                return

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
                    sample['obs'] = np.concatenate([sample['obs'].flatten(), [sample['acts']]]) # For now hard code 
                    samples_queue.put(sample)
            except (IndexError, EOFError, BrokenPipeError) as e:
                # print(f'Error {e} accessing index {index} in writer worker {worker_id}. Length of obs {len(self.all_obs)}')
                pass
            except Exception as e:
                pass
            # time.sleep(0.01)
                        
        # samples_queue.put(None)

def process_batch(batch, batch_number, obs_list, acts_list):
    with torch.no_grad():
        obs = batch['obs']
        acts = batch['acts']
        obs_list.extend(obs.cpu().numpy())
        acts_list.extend(acts.cpu().numpy().astype(int))

# Define a function to calculate a portion of the sample_counts
def calculate_sample_counts(col_range, obs_list, feature_ranges):
    col_start, col_end = col_range
    partial_counts = []
    for col in range(col_start, col_end):
        col_counts = [((obs_list[:, col] > feature_ranges[i]) & (obs_list[:, col] <= feature_ranges[i + 1])).sum()
                    for i in range(len(feature_ranges) - 1)]
        partial_counts.append(col_counts)
    return partial_counts


def analyze_data(zip_filename, obs_list, acts_list, **kwargs):
    n_cpu=kwargs['n_cpu']
    val_batch_count=kwargs['val_batch_count']
    # train_data_loader                                             = create_dataloaders(
    #                                                                                               zip_filename,
    #                                                                                               train_datasets=[], 
    #                                                                                               type = 'train',
    #                                                                                               device=device,
    #                                                                                               batch_size=minibatch_size,
    #                                                                                               n_cpu = n_cpu,
    #                                                                                               visited_data_files=set([])
    #                                                                                           )
    train_data_loader                                             =  CustomDataLoader(
                                                                        zip_filename, 
                                                                        device=kwargs['device'],
                                                                        batch_size=kwargs['batch_size'],
                                                                        n_cpu=n_cpu,
                                                                        val_batch_count=val_batch_count,
                                                                        chunk_size=kwargs['chunk_size'],
                                                                        type= kwargs['type'],
                                                                        plot_path=kwargs['plot_path'],
                                                                        validation=kwargs['validation'],
                                                                        visited_data_files = set([])
                                                                    ) 
    train_data_loader_iterator = iter(train_data_loader)
    # Create a DataFrame from the data loader
    data_list = np.empty((0, 101))

    num_processes = n_cpu

    with multiprocessing.Pool(processes=num_processes) as pool:
        for batch_number in range(val_batch_count):
            try:
                batch = next(train_data_loader_iterator)
                pool.apply_async(process_batch, (batch, batch_number, obs_list, acts_list))
            except StopIteration:
                print('StopIteration Error ')
                break
        
        pool.close()
        pool.join()

    obs_list = np.array(obs_list)
    acts_list = np.array(acts_list)

    actions = np.array(acts_list)
    action_counts = np.bincount(actions.astype(int))
    print('action_counts ', action_counts)

    # Create a bar chart for action distribution
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(action_counts)), action_counts, tick_label=range(len(action_counts)))
    plt.xlabel("Action")
    plt.ylabel("Frequency")
    plt.title(f"Action Distribution for {zip_filename}")
    plt.savefig("Action_Distribution.png")
    # plt.show()

    # Define the ranges for each feature
    feature_ranges = np.linspace(-1, 1, num=101)  # Adjust the number of bins as needed


    

    col_ranges = [(col, col + 1) for col in range(obs_list.shape[1])]

    # Create a partially-applied function with fixed arguments
    calculate_sample_counts_partially_wrapped = functools.partial(calculate_sample_counts, obs_list=obs_list, feature_ranges=feature_ranges)


    with multiprocessing.Pool(processes=num_processes) as pool:
        partial_counts_list = pool.map(calculate_sample_counts_partially_wrapped, col_ranges)

    # Concatenate the partial counts to obtain the complete sample_counts
    sample_counts = np.concatenate(partial_counts_list, axis=0)
    
    sample_counts = sample_counts.T
    print('Sample counts done')
            
    # Normalize the sample counts to a range between 0 and 1
    normalized_counts = sample_counts / sample_counts[0,:].sum()
    # Reshape the sample counts to create a heatmap
    # sample_count_matrix = sample_counts.reshape(-1, 1)
    

    if 'plot' in kwargs and kwargs['plot']:
        # Create a color map based on the sample counts
        cmap = sns.cubehelix_palette(start=2, rot=0, dark=0, light=1.0, reverse=False, as_cmap=True)


        # Create a violin plot directly from the data loader
        # Create a figure with two subplots
        fig, axes = plt.subplots(2, 1, figsize=(12, 6))
        sns.violinplot(data=obs_list , inner="quartile", ax=axes[0])
        axes[0].set_title(f"Violin Plot (input) for {zip_filename}")
        sns.heatmap(data=sample_counts, cmap=cmap, cbar=False, ax=axes[1])
        axes[1].set_title(f"Heatmap (input) for {zip_filename}")

        plt.tight_layout()
        plt.show()
        plt.savefig('Analysis.png')
        print('Plotting done')
    return normalized_counts

def validation(policy, device, project, zip_filenames, batch_size, minibatch_size, n_cpu ,visited_data_files, val_batch_count = 2500):

    zip_filenames = zip_filenames if isinstance(zip_filenames, list) else [zip_filenames]
    val_device = torch.device('cpu')
    policy.to(val_device)
    policy.eval()
    type = 'val'
    with torch.no_grad():
        # val_data_loader                                             =  CustomDataLoader(
        #                                                                                 zip_filename, 
        #                                                                                 device=val_device,
        #                                                                                 batch_size=batch_size,
        #                                                                                 n_cpu=n_cpu,
        #                                                                                 val_batch_count=val_batch_count,
        #                                                                                 chunk_size=500,
        #                                                                                 type= type,
        #                                                                                 plot_path=None,
        #                                                                                 visited_data_files = set([])
        #                                                                             ) 
        train_datasets = []
        for zip_filename in zip_filenames:
            val_data_loader =                                                       create_dataloaders(
                                                                                                        zip_filename,
                                                                                                        train_datasets = train_datasets, 
                                                                                                        type = type,
                                                                                                        device=device,
                                                                                                        batch_size=minibatch_size,
                                                                                                        n_cpu = n_cpu,
                                                                                                        visited_data_files= visited_data_files 
                                                                                                    )
        metrics                      = calculate_validation_metrics(
                                                                        val_data_loader,
                                                                        policy, 
                                                                        zip_filename=zip_filename,
                                                                        device=val_device,
                                                                        batch_size=batch_size,
                                                                        n_cpu=n_cpu,
                                                                        val_batch_count=val_batch_count,
                                                                        chunk_size=500,
                                                                        type= type,
                                                                        validation = True,
                                                                        plot_path=None
                                                                    )