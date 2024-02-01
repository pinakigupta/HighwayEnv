import torch.nn as nn
from torch import multiprocessing
import os, shutil
os.environ["HDF5_USE_THREADING"] = "true"
import h5py
import sys
from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy
import torch
import pygame  
import numpy as np
import seaborn as sns
from highway_env.utils import lmap
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
import zipfile
import time
import signal
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from contextlib import ExitStack
from gymnasium.spaces import Space as Space
import numpy as np
import tracemalloc

from highway_env.envs.common.action import DiscreteMetaAction
ACTIONS_LAT = DiscreteMetaAction.ACTIONS_LAT

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
           
            
# def list_of_dicts_to_dict_of_lists(dict_):
#     # dict_, obs_list, acts_list = arg   
#     obs_list = []
#     acts_list = []  
#     for key, value in dict_.items():
#         if key == 'obs':
#             obs_list.append(value)
#         if key == 'acts':
#             acts_list.append(value)
#     return obs_list, acts_list


'''
    Workers trying to process input data queue and put validation metrices on the output queue
'''

def validation_batch_worker_process(output_queue, input_queue, labels, worker_index,lock, batch_count=None, **kwargs):
    # local_policy_path = f'/tmp/validation_policy{worker_index}.pth'
    # shutil.copy('validation_policy.pth', local_policy_path)
    if lock.acquire(timeout=10):
        local_policy = torch.load('validation_policy.pth', map_location='cpu')
        lock.release()
    else:
        print(f"Couldn't fetch the validation policy for worker {worker_index}", flush = True)
        return {}
    
    output_samples =[]
    while True:
        try:
            if not input_queue.empty():
                batch = input_queue.get()
            else:
                time.sleep(0.1)
                # print(f"Input queue empty. Output queue size {output_queue.qsize()} ", flush=True)
                continue
            # print(f"batch collected from worker {worker_index}")
            true_labels = batch['acts'].numpy()
            with torch.no_grad():

                predicted_labels = [local_policy.predict(obs.numpy())[0] for obs in batch['obs']]
                # predicted_labels = true_labels
                _, log_prob, entropy = local_policy.evaluate_actions(batch['obs'], batch['acts'])
                

                if 'label_weights' in kwargs:
                    true_labels         = true_labels @ kwargs['label_weights']
                    predicted_labels    = predicted_labels @ kwargs['label_weights']
                # print(f"Fetched the validation policy for worker {worker_index}. predicted_labels {predicted_labels}", flush = True)
                accuracy = accuracy_score(true_labels, predicted_labels)
                # print(f"Lock acquired by {worker_index}. True labels {true_labels}. \n predicted_labels {predicted_labels}. \n accuracy {accuracy}")
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
                # print( predicted_labels , " output_sample ", output_sample, flush=True)
                output_queue.put(output_sample)
            # if lock.acquire(timeout=0.1):
            #     output_queue.put(output_samples)
            #     lock.release()
            # else:
            #     output_samples.append(output_sample)
        except Exception as e:
            print(f" Error in validation_batch_worker_process {e}")
            raise(e)


'''
    Use individual workers to collect samples of validation metrics data and combine them to create the final validation metrics
'''
def calculate_validation_metrics(manager, val_data_loader, policy,zip_filename, **kwargs):
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

    def calculate_metrics(output_queue):
        nonlocal accuracies, precisions, recalls, f1s, cross_entropies, entropies
        result = output_queue.get()
        accuracies.append(result['accuracy'])
        precisions.append(result['precision'])
        recalls.append(result['recall'])
        f1s.append(result['f1'])
        cross_entropies.append(result['cross_entropy'])
        entropies.append(result['entropy'])
        # if conf_matrix.size == 0 :
        #     conf_matrix = result['conf_matrix']
        # else:
        #     conf_matrix += result['conf_matrix']
    # manager = multiprocessing.Manager()
    # with manager:
    if True:
        lock = manager.Lock()
        torch.save(policy,'validation_policy.pth')
        # Create a list to store process objects
        processes = []
        output_queue =  manager.Queue()
        input_queue =   manager.Queue()
        num_workers =   max(kwargs['n_cpu'],1)

        for i in range(num_workers):
            
            worker_process = multiprocessing.Process(
                                                        target=validation_batch_worker_process, 
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

        val_batch_count = kwargs['val_batch_count']
        progress_bar = tqdm(total=val_batch_count, desc= f'{kwargs["type"]} progress')
        conf_matrix = np.array([])        
        batch_count = 0

        
        
    # with multiprocessing.Manager() as manager:
        # managed_accuracy        = manager.list()
        # managed_precision       = manager.list()
        # managed_recall          = manager.list()
        # managed_f1              = manager.list()
        # managed_cross_entropy   = manager.list()
        # managed_entropy         = manager.list()   
        for batch in val_data_loader:
            sample = {key: value.to(torch.device('cpu')) for key, value in batch.items()}
            try:
                input_queue.put(sample)
                if input_queue.qsize() > 1.25*val_batch_count:
                    break
                if len(accuracies) > val_batch_count:
                    break
            except Exception as e:
                pass
                # print(e)
            # print(f"batch_count {batch_count}")
            batch_count += 1
            if output_queue.empty():
                continue
            calculate_metrics(output_queue)
            progress_bar.update(1)
            
            
        val_batch_count = min(val_batch_count, batch_count )
        print('val_batch_count ', val_batch_count, ' input_queue size ', input_queue.qsize(), ' output queue size ', output_queue.qsize())
            

        while len(accuracies) < 0.9*val_batch_count:
            # if output_queue.empty() and input_queue.empty():
            #     break
            # el
            if output_queue.empty():
                time.sleep(0.1)
                continue
            calculate_metrics(output_queue)
            progress_bar.update(1)

        progress_bar.close()



        # Wait for all processes to finish
        for process in processes:
            # print(f'terminating process {process.pid}')
            process.terminate()

                
    # print('Calculate evaluation metrics')
    # Calculate evaluation metrics
    axis = 0
    accuracy  = np.mean(accuracies,  axis=axis) 
    precision = np.mean(precisions, axis=axis)
    recall    = np.mean(recalls, axis=axis)
    f1        = np.mean(f1s, axis=axis)
    cross_entropy = np.mean(cross_entropies)
    entropies = np.mean(entropies)

    # Print the metrics
    print("Accuracy:", accuracy, np.mean(accuracy))
    print("cross_entropy:", np.mean(cross_entropy), "entropy:", np.mean(entropies))
    print("Precision:", precision, np.mean(precision))
    print("Recall:", recall, np.mean(recall))
    print("F1 Score:", f1, np.mean(f1))



    if conf_matrix:
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
        # print([worker.pid for worker in processes if worker.is_alive()])

    metrics = {
               'accuracy':      accuracy,
               'precision':     np.mean(precision), 
               'recall':        np.mean(recall), 
               'f1':            np.mean(f1),
               'cross_entropy': np.mean(cross_entropy),
               'entropy':       np.mean(entropies),

            }
    return metrics




    
