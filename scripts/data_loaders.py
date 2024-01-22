from torch import multiprocessing
import os
os.environ["HDF5_USE_THREADING"] = "true"
from torch.utils.data import Dataset, DataLoader
import torch
from generate_expert_data import extract_post_processed_expert_data
import numpy as np
import seaborn as sns
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import zipfile
import time
import signal
from torch.utils.data import DataLoader, ConcatDataset, Subset
import random
import math
import functools
from collections import defaultdict, Counter
from gymnasium.spaces import Space as Space
import numpy as np

from utilities import *

class CustomDataset(Dataset):
    def __init__(self, data, device, pad_value=0, **kwargs):
        # Load your data from the file and prepare it here
        # self.data = ...  # Load your data into this variable
        self.keys_attributes = kwargs['keys_attributes']
        if isinstance(data, dict) or isinstance(data, list):
            self.data =  data
        else:
            self.data_file = data
            self.data = extract_post_processed_expert_data(self.data_file, self.keys_attributes)
        self.pad_value = pad_value
        self.device = device
        # print(" data lengths ", len(self.exp_obs), len(self.exp_acts), len(self.exp_dones))
        return

    def __len__(self):
        # print("data length for custom data set ",id(self), " is ", len(self.exp_acts),  len(self.exp_obs))
        try:
            self.length = len(self.data['acts'])
        except Exception as e:
            self.length = len(self.data)
        return self.length

    def __getitem__(self, idx):
        # idx = idx % self.__len__()
        sample = {}
        prev_sample = {}
        try:
            for key in self.keys_attributes:
                if key in self.data or key in self.data[0]:
                    try:
                        # Attempt to deserialize the data as JSON
                        value = json.loads(self.data[key][idx])
                        prev_value =  json.loads(self.data[key][idx-1]) if idx>1 else value
                        # If JSON deserialization fails, assume it's a primitive type
                        # print(f"data key{key}, index{idx} of length {len(self.data[key])}")
                    except Exception as e:
                        try:
                            value = self.data[key][idx]
                            prev_value = self.data[key][idx-1] if idx>1 else value
                        except Exception as e:
                            try:                        
                                value = self.data[idx][key]
                                prev_value = self.data[idx-1][key] if idx>1 else value
                            except Exception as e:
                                print(e)
                                print(' idx ', idx, ' data length ', len(self.data))
                                return sample
                                # raise(e)
                    sample[key] = torch.tensor(value, dtype=torch.float32).to(self.device)
                    prev_sample[key] = torch.tensor(prev_value, dtype=torch.float32).to(self.device)

        except Exception as e:
            print(f' {e} , for , { id(self)}. key {key} , value {value} ')
            pass
            # raise e

        try:
            sample['obs'][:, -7].fill_(0)
            prev_sample = prev_sample['acts'].view(1) if 'acts' in prev_sample else prev_sample['act'].view(1)
            sample['obs'] = torch.cat([ sample['obs'].view(-1), prev_sample], dim=0)
        except Exception as e:
            pass
        # sample['obs'] = {'obs':sample['obs'], 'action':prev_sample['act']}

        return sample 


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
    
def process_sample_for_balanced_subset(args):
    sample, class_distribution, total_scanned_samples = args
    if sample:
        acts_value = sample['acts'].item()
        # with class_distribution.get_lock():
        if acts_value in class_distribution:
            class_distribution[acts_value] += 1
        else:
            class_distribution[acts_value] = 1
        total_scanned_samples.value += 1


def create_balanced_subset(dataset, shuffled_indices, class_distribution = None , alpha = 9.1):
    
    class_distribution = class_distribution if class_distribution else Counter(sample['acts'].item() for sample in dataset)
    
    # Calculate min_samples_per_class as the actual minimum count in the dataset
    min_samples_per_class = int(min(class_distribution.values()))

    # Calculate max_samples_per_class based on min_samples_per_class and alpha
    max_samples_per_class = int(min_samples_per_class * alpha)

    balanced_indices = []
    class_counts = {}
    
    for index in shuffled_indices:
        label = int(dataset[index]['acts'].item())
        
        if label in class_counts:
            if class_counts[label] < max_samples_per_class:
                balanced_indices.append(index)
                class_counts[label] += 1
        else:
            balanced_indices.append(index)
            class_counts[label] = 1

    print(f"After shuffling and downsampling class_counts {class_counts}")
    # Create the Subset
    balanced_subset = Subset(dataset, balanced_indices)

    return balanced_subset

def load_data_for_single_file_within_a_zipfile(args):    
    zip_filename, train_data_file, visited_data_files, device, val_only = args  
    visited_filepath = zip_filename + train_data_file
    if visited_filepath not in visited_data_files:
        visited_data_files.append(visited_filepath)
        with zipfile.ZipFile(zip_filename, 'r') as zipf:
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

                class_distribution = Counter(sample['acts'].item() for sample in modified_dataset)
                print('modified_dataset', class_distribution)

                shuffled_indices = np.arange(len(modified_dataset))
                balanced_modified_dataset = None
                if val_only:
                    return modified_dataset
                else:
                    balanced_modified_dataset = create_balanced_subset(modified_dataset, shuffled_indices,  alpha=3.1)
                    balanced_modified_dataset_list = [{key: value.cpu().numpy() for key, value in balanced_modified_dataset[i].items()}  for i in range(len(balanced_modified_dataset))]
                    return balanced_modified_dataset_list
                        
                print(f"Dataset appended for  {train_data_file}")
                
                # del balanced_modified_dataset
                # del samples
                # del modified_dataset

                 
def create_dataloaders(args):
    zip_filename, visited_data_files_list, device, kwargs = args
    val_only = kwargs['type'] == 'val'
    # n_cpu = kwargs['n_cpu']
    # Extract the names of the HDF5 files from the zip archive
    with zipfile.ZipFile(zip_filename, 'r') as zipf:
        print(f" File handle for the zip file {zip_filename} opened ")
        hdf5_train_file_names = [file_name for file_name in zipf.namelist() if file_name.endswith('.h5')  and kwargs['type'] in file_name]

    train_data_list = []
    with multiprocessing.Manager() as manager:
        visited_data_files  = manager.list(visited_data_files_list)
        # managed_train_data_list     = manager.list()        
        with multiprocessing.get_context('spawn').Pool(processes=kwargs['n_cpu']) as pool:
            pool_args = [ (zip_filename, train_data_file, visited_data_files, device, val_only) for train_data_file in hdf5_train_file_names]
            results = pool.map(load_data_for_single_file_within_a_zipfile, pool_args)
            
        for result in results:
            train_data_list.extend(result)
            
        visited_data_files_list = list(visited_data_files)
        # train_data_list = list(managed_train_data_list) 
        print(' managed_train_data_list len ', len(train_data_list))
    return train_data_list

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

def validation(policy, device, zip_filenames, batch_size, minibatch_size, n_cpu ,visited_data_files, val_batch_count = 2500):

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
        # train_datasets = []
        val_data_loader = multiprocess_data_loader(zip_filenames, visited_data_files , device , minibatch_size, type = 'train', n_cpu = n_cpu)
        metrics                      = calculate_validation_metrics(
                                                                        val_data_loader,
                                                                        policy, 
                                                                        zip_filename=zip_filenames,
                                                                        device=val_device,
                                                                        batch_size=batch_size,
                                                                        n_cpu=n_cpu,
                                                                        val_batch_count=val_batch_count,
                                                                        chunk_size=500,
                                                                        type= type,
                                                                        validation = True,
                                                                        plot_path=None
                                                                    )
        
def multiprocess_data_loader(zip_filenames, visited_data_files_list , device , minibatch_size, type = 'train', **kwargs):
    
    list_of_dicts = []
    for zip_filename in zip_filenames:
        args = (zip_filename, visited_data_files_list , device, {'type': type, 'n_cpu': kwargs['n_cpu']})
        list_of_dicts.extend(create_dataloaders(args))
    print("All datasets appended. dataset lenght " , len(list_of_dicts), ' keys ', list_of_dicts[0].keys())
        

    shuffled_indices = np.arange(len(list_of_dicts))
    np.random.shuffle(shuffled_indices)
    print(' dict_of_lists compiled. Length is ', len(shuffled_indices), max(shuffled_indices), min(shuffled_indices))
    
    shuffled_combined_train_dataset = create_balanced_subset(list_of_dicts, shuffled_indices)
    print(f'shuffled_combined_train_dataset distribution {Counter(sample["acts"].item() for sample in shuffled_combined_train_dataset)}')
    print(f'Total batch count in data set {len(shuffled_combined_train_dataset)//minibatch_size}')
    _data_loader = DataLoader(
                                    shuffled_combined_train_dataset, 
                                    batch_size=minibatch_size, 
                                    shuffle=True,
                                    # sampler=sampler,
                                    drop_last=True,
                                    num_workers=5,
                                    # pin_memory=True,
                                    # pin_memory_device=device,
                                    )
    return _data_loader


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
            # print(batch_samples[0].keys())
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
