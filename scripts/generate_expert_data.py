import numpy as np
import torch
from tqdm import tqdm
from torch.nn import Module
from torch import multiprocessing
import copy
from torch import multiprocessing as mp
import h5py
import gymnasium as gym
from collections import Counter
from scipy.stats import entropy
import random
import wandb
import json
import zipfile
import os

from forward_simulation import append_key_to_dict_of_dict

torch.set_default_tensor_type(torch.FloatTensor)

def make_configure_env(**kwargs):
    env = gym.make(kwargs["id"], render_mode=kwargs["render_mode"], config=kwargs["config"])
    # env.configure(kwargs["config"])
    env.reset()
    return env


def worker(
                placeholder, 
                data_queue, 
                episode_rewards, 
                steps_per_worker, 
                # processes_launched, 
                worker_id,
                # progress,
                lock,
                **env_kwargs 
                ):
    env = make_configure_env(**env_kwargs).unwrapped #env_value.value  # Retrieve the shared env object
    steps_collected = 0
    # oracle = env.controlled_vehicles[0]

    oracle = torch.load('oracle.pth', map_location='cpu')

    while True:
        ob = env.reset()
        ob=ob[0]
        done = False
        ep_rwds = []
        all_obs = {}
        all_acts = {}
        all_done = {}
        rollout_steps = 0
        experts_to_consider = []
        if('expert' in env_kwargs) and (env_kwargs['expert']=='MDPVehicle'):
            experts_to_consider = [env.vehicle]
        else:
            for v in env.road.vehicles:
                if v is not env.vehicle:
                    experts_to_consider.append(v) 
        # print(" Entered worker ", worker_id, " . num_steps ", steps_per_worker,  flush=True)
        while rollout_steps < steps_per_worker:
            # Extract features from observations using the feature extractor
            # features_extractor, policy_net, action_net = oracle
            ob_tensor = torch.Tensor(ob).detach().to(torch.device('cpu'))


            act = oracle.predict(ob)[0]
            # act = 4
            next_ob, rwd, done, _, _ = env.step(act)
            rollout_steps += 1


            for v in experts_to_consider:
                obs = v.observer
                discrete_action = v.discrete_action()
                acts = env.action_type.actions_indexes[discrete_action]
                if v not in all_obs:
                    all_obs[v] = []
                    all_acts[v] = []
                    all_done[v] = []
                all_obs[v].append(obs)
                all_acts[v].append(acts)
                all_done[v].append(False)
            # print(type(all_done[-1]), len(all_done[-1]))
            
            # print(id(env.vehicle), " all_obs ", len(all_obs[env.vehicle]), discrete_action, done)
            if done:
                for v in all_done:
                    all_done[v][-1] = True
                break
            ob = next_ob
            # data_queue.put((ob, act))
            ep_rwds.append(rwd)
            # steps_collected += obs_collected
        # print("all_obs ", len(all_obs[env.vehicle]), len(all_acts[env.vehicle]), len(all_done[env.vehicle]))
        # Update progress value
        if lock.acquire(timeout=1):
            for (v , ep_obs), (v_, ep_acts), (v_, ep_done) in zip(all_obs.items(), all_acts.items(), all_done.items()):
                # print(" ep_acts length ", len(ep_acts), id1, id2, " steps_collected ", steps_collected, 
                #       " obs_collected ", obs_collected, " steps_per_worker ", steps_per_worker, " done ", done)
                # if v.crashed:
                #     pass
                #     # print(" Discarding data as vehicle ", v, " crashed " )
                # else:
                # print("ep_obs ", ep_obs)
                data_queue.put((ep_obs, ep_acts, ep_done))
                steps_collected += len(ep_acts)
            lock.release()
        else:
            print("lock time out occurred")
            # progress.value += obs_collecected
            # print("worker ", worker_id, " steps_collected ", steps_collected,   flush=True)
        # episode_rewards.append(np.sum(ep_rwds))
        # time.sleep(0.001)

def collect_expert_data(
                        oracle,
                        num_steps_per_iter,
                        train_filename,
                        validation_filename,
                        train_ratio = 0.8,
                        **env_kwargs,
                        ):
    
    if os.path.exists(train_filename):
        os.remove(train_filename)
    if os.path.exists(validation_filename):
        os.remove(validation_filename)
    # with h5py.File(filename, 'w') as hf:
    #         pass  # This just opens and immediately closes the file, effectively clearing it


    torch.set_num_threads(1)
    # Create the shared Manager object
    manager = torch.multiprocessing.Manager()
    # env_value = manager.Value(type(None), None)


    # Create a lock for workers
    lock = mp.Lock()
    
    processes_launched = multiprocessing.Event()

    # Initialize a queue to store oracle data
    exp_data_queue = multiprocessing.Queue()

    # Initialize a list to store episode rewards
    episode_rewards = manager.list()

    # Determine the number of workers based on available CPU cores
    num_workers =  max(multiprocessing.cpu_count()-3,1)

    # Calculate the number of steps per worker
    num_steps_per_worker = min(num_steps_per_iter , 2 * (num_steps_per_iter// num_workers))
    # num_steps_per_worker *=1.25 # allocate higher number of episodes than quota, so that free workers can do them w/o a blocking call

    # Create a list to store worker processes
    worker_processes = []

    # env_value.value = env
    # Launch worker processes for oracle data collection
    oracle.to('cpu')
    torch.save(oracle,'oracle.pth')
    for i in range(num_workers):
        
        worker_process = multiprocessing.Process(
                                                    target=worker, 
                                                    args=(
                                                            oracle, 
                                                            exp_data_queue, 
                                                            episode_rewards, 
                                                            num_steps_per_worker, 
                                                            # processes_launched, 
                                                            i,
                                                            # progress,
                                                            lock
                                                        ),
                                                    kwargs = env_kwargs, 
                                                )
        worker_processes.append(worker_process)
        worker_process.start()


    pbar_outer = tqdm(total=num_steps_per_iter, desc='Progress of Expert data collection')

    # Collect oracle data from the queue
    ep_collected = 0
    steps_count = 0
    exp_acts = []
    

    with h5py.File(train_filename, 'a') as train_hf, h5py.File(validation_filename, 'a') as valid_hf:
        while steps_count < 0.9*num_steps_per_iter:
            ob, act, done = exp_data_queue.get()
            exp_acts_temp =copy.deepcopy(exp_acts)
            exp_acts_temp.extend(act)


            # if is_skewed(
            #                 act,
            #                 threshold=0.5
            #             ):
            #     # print("discarding episode as it is making the data skewed")
            #     continue

            exp_acts = exp_acts_temp
            steps_count += len(act)
            ep_collected +=1
                    # Stop collecting data
            
            # Save the data list as an HDF5 file
            if random.random() < train_ratio:
                hf = train_hf
            else:
                hf = valid_hf

            episode_group = hf.create_group(f'episode_{ep_collected}')
            for i, (arr1, arr2, arr3) in enumerate(zip(ob, act, done)):
                episode_group.create_dataset(f'exp_obs{i}',  data=arr1,  dtype='float32')
                episode_group.create_dataset(f'exp_acts{i}', data=arr2,  dtype='float32')
                episode_group.create_dataset(f'exp_done{i}', data=arr3,  dtype=bool)
            pbar_outer.update(steps_count - pbar_outer.n)

    # print(" joining worker processes ", [worker.pid for worker in worker_processes], flush=True)


    # Join worker processes to wait for their completion
    for worker_process in worker_processes:
        worker_process.terminate()

    # print(" End of worker_process join")

    # Close and join the queue
    exp_data_queue.close()
    # print(" End of data queue")
    exp_data_queue.join_thread()
    pbar_outer.close()



def postprocess(inputfile,outputfile):
    exp_obs, exp_acts, exp_dones = extract_expert_data(inputfile)
    class_distribution = Counter(exp_acts)
    # print(" Before post process class_distribution ", class_distribution, ' len(exp_obs) ' , len(exp_obs))
    exp_obs, exp_acts, exp_dones = downsample_most_dominant_class(exp_obs, exp_acts, exp_dones, factor=1.25)
    class_distribution = Counter(exp_acts)
    # print(" After post process class_distribution ", class_distribution, ' len(exp_obs) ' , len(exp_obs))
    # Convert the list of arrays into a NumPy array
    numpy_exp_obs = np.array(exp_obs)
    numpy_exp_acts = np.array(exp_acts)
    numpy_exp_dones = np.array(exp_dones)
    with h5py.File(outputfile, 'w') as hf:
        hf.create_dataset('obs',  data=numpy_exp_obs)
        hf.create_dataset('act',  data=numpy_exp_acts)
        hf.create_dataset('dones',  data=numpy_exp_dones)

def downsample_most_dominant_class(exp_obs, exp_acts, exp_dones, factor=2.0):
    # Calculate the distribution of classes
    class_distribution = Counter(exp_acts)
    actions_indexes = {'LANE_LEFT': 0, 'IDLE': 1, 'LANE_RIGHT': 2, 'FASTER': 3, 'SLOWER': 4}
    # Initialize all classes to 0 in the class_distribution
    for action_index in actions_indexes.values():
        class_distribution[action_index] = class_distribution.get(action_index, 0)
    most_common_class, most_common_count = class_distribution.most_common(1)[0]
    second_most_common_class, second_most_common_count = class_distribution.most_common(2)[1]
    if second_most_common_count == 0:
        return exp_obs, exp_acts, exp_dones
    desired_samples = factor * second_most_common_count
    # Determine whether downsampling is needed
    if most_common_count > factor * second_most_common_count:
        # Calculate the desired number of samples for downsampling
        

        # Perform downsampling on the most common class
        downsampled_exp_obs = []
        downsampled_exp_acts = []
        downsampled_exp_dones = []

        for ob, act, done in zip(exp_obs, exp_acts, exp_dones):
            # print(class_distribution, desired_samples)
            if act == most_common_class and class_distribution[act] > desired_samples:
                class_distribution[act] -= 1
                continue

            else:
                downsampled_exp_obs.append(ob)
                downsampled_exp_acts.append(act)
                downsampled_exp_dones.append(done)

      # Upsample less dominant classes by copying their indices
        upsampled_exp_obs = []
        upsampled_exp_acts = []
        upsampled_exp_dones = []

        for act, count in class_distribution.items():
            if act != most_common_class:
                num_samples_to_copy = int(second_most_common_count - count)
                indices_to_copy = [i for i, a in enumerate(exp_acts) if a == act]
                for index in indices_to_copy[:num_samples_to_copy]:
                    upsampled_exp_obs.append(exp_obs[index])
                    upsampled_exp_acts.append(exp_acts[index])
                    upsampled_exp_dones.append(exp_dones[index])

        final_exp_obs = downsampled_exp_obs + upsampled_exp_obs
        final_exp_acts = downsampled_exp_acts + upsampled_exp_acts
        final_exp_dones = downsampled_exp_dones + upsampled_exp_dones

        return final_exp_obs, final_exp_acts, final_exp_dones
    else:
        # No downsampling needed, return original data
        return exp_obs, exp_acts, exp_dones

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def normalize_distribution(distribution, num_action_types):
    total_samples = sum(distribution.values())
    normalized_distribution = {action: count / total_samples for action, count in distribution.items()}
    # Ensure that the normalized distribution includes all possible action types
    for action in range(num_action_types):
        if action not in normalized_distribution:
            normalized_distribution[action] = 0.0
    return normalized_distribution

def is_skewed(actual_distribution, threshold, num_action_types=5):
    uniform_normalized_distribution = [1 / num_action_types] * num_action_types
    actual_normalized_distribution = normalize_distribution(Counter(actual_distribution), num_action_types)
    kl_divergence = entropy(list(actual_normalized_distribution.values()), uniform_normalized_distribution)
    return kl_divergence > threshold


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
                try:
                    if dataset_name.startswith('exp_obs'):
                        exp_obs.extend([dataset[:]])
                    elif dataset_name.startswith('exp_acts'):
                        exp_acts.extend([dataset[()]])
                    elif dataset_name.startswith('exp_done'):
                        exp_done.extend([dataset[()]])
                except Exception as e:
                    pass
                    # print(e)
           

    return  exp_obs, exp_acts, exp_done


def extract_post_processed_expert_data(filename):
    # Open the HDF5 file in read mode
    with h5py.File(filename, 'r') as hf:
        # Read the 'obs' dataset
        obs_array = hf['obs'][:]  # [:] to read the entire dataset
        
        # If you've saved other datasets (e.g., 'act' and 'done'), you can read them similarly
        act_array = hf['act'][:]
        done_array = hf['dones'][:]

    return obs_array, act_array, done_array

def retrieve_agent( artifact_version, agent_model ,project = None):
    # Initialize wandb
    wandb.init(project=project, name="inference")
    # Access the run containing the logged artifact

    # Download the artifact
    artifact = wandb.use_artifact(artifact_version)
    artifact_dir = artifact.download()
    wandb.finish()

    # Load the model from the downloaded artifact
    optimal_gail_agent_path = os.path.join(artifact_dir, agent_model) #, "optimal_gail_agent.pth")
    # final_gail_agent_path = os.path.join(artifact_dir, "final_gail_agent.pth")

    # final_gail_agent = torch.load(final_gail_agent_path)
    optimal_gail_agent = torch.load(optimal_gail_agent_path)
    return optimal_gail_agent

def expert_data_collector(
                            oracle_agent,
                            data_folder_path,
                            zip_filename,
                            **env_kwargs
                            ):
    with open("config.json") as f:
        config = json.load(f)
    expert_temp_data_file=f'expert_T_data_.h5'
    validation_temp_data_file = f'expert_V_data_.h5'
    device = torch.device("cpu")    
    append_key_to_dict_of_dict(env_kwargs,'config','duration',20)
    with zipfile.ZipFile(zip_filename, 'a') as zipf:
        # Create an outer tqdm progress bar
        highest_filenum = max(
                                (
                                    int(filename.split('_')[3].split('.')[0])
                                    for filename in os.listdir(data_folder_path)
                                    if filename.startswith('expert_train_data_') and filename.endswith('.h5')
                                    and filename.split('_')[3].split('.')[0].isdigit()
                                ),
                                default=-1
                                )
        total = 0
        if 'total_iterations' in env_kwargs:
            total_iterations = env_kwargs['total_iterations']  # The total number of loop iterations
            total = total_iterations
        elif 'delta_iterations' in env_kwargs:
            total = env_kwargs['delta_iterations']
            total_iterations = highest_filenum + total +1 
        outer_bar = tqdm(total=total, desc="Outer Loop Progress")
        print("highest_filenum ", highest_filenum)
        for filenum in range(highest_filenum+1, total_iterations):
            collect_expert_data  (
                                        oracle=oracle_agent,
                                        num_steps_per_iter=config["num_expert_steps"],
                                        train_filename=expert_temp_data_file,
                                        validation_filename=validation_temp_data_file,
                                        **env_kwargs
                                    )
            # print("collect data complete")
            exp_file = f'expert_train_data_{filenum}.h5'
            val_file = f'expert_val_data_{filenum}.h5'
            postprocess(expert_temp_data_file, exp_file)
            postprocess(validation_temp_data_file, val_file)
            zipf.write(exp_file)
            zipf.write(val_file)
            outer_bar.update(1)
        outer_bar.close()