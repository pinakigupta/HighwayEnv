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


torch.set_default_tensor_type(torch.FloatTensor)

def make_configure_env(**kwargs):
    env = gym.make(kwargs["id"], render_mode=kwargs["render_mode"], config=kwargs["config"])
    # env.configure(kwargs["config"])
    env.reset()
    return env


def worker(
                expert, 
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
    # expert = env.controlled_vehicles[0]

    while steps_collected < steps_per_worker:
        ob = env.reset()
        ob=ob[0]
        done = False
        ep_rwds = []
        all_obs = {}
        all_acts = {}
        all_done = {}

        # print(" Entered worker ", worker_id, " . num_steps ", steps_per_worker,  flush=True)
        while steps_collected < steps_per_worker:
            # Extract features from observations using the feature extractor
            # features_extractor, policy_net, action_net = expert
            ob_tensor = copy.deepcopy(torch.Tensor(ob).to(torch.device('cpu')))

            
            act = expert.predict(ob_tensor)[0]
            # act = 0 
            next_ob, rwd, done, _, _ = env.step(act)


            for v in env.road.vehicles:
                if v is not env.vehicle: 
                    obs = v.observer
                    acts = env.action_type.actions_indexes[v.discrete_action]
                    if v not in all_obs:
                        all_obs[v] = []
                        all_acts[v] = []
                        all_done[v] = []
                    all_obs[v].append(obs)
                    all_acts[v].append(acts)
                    all_done[v].append(False)
            # print(type(all_done[-1]), len(all_done[-1]))
            if done:
                for v in all_done:
                    all_done[v][-1] = True
                break

            ob = next_ob
            # data_queue.put((ob, act))
            ep_rwds.append(rwd)
            # steps_collected += obs_collected

        # print("all_obs ", len(all_obs))
        # Update progress value
        if lock.acquire(timeout=1):
            for (v , ep_obs), (v_, ep_acts), (v_, ep_done) in zip(all_obs.items(), all_acts.items(), all_done.items()):
                # print(" ep_acts length ", len(ep_acts), id1, id2, " steps_collected ", steps_collected, 
                #       " obs_collected ", obs_collected, " steps_per_worker ", steps_per_worker, " done ", done)
                if v.crashed:
                    pass
                    # print(" Discarding data as vehicle ", v, " crashed " )
                else:
                    data_queue.put((ep_obs, ep_acts, ep_done))
                    steps_collected += len(ep_acts)
            lock.release()
            # progress.value += obs_collecected
            # print("worker ", worker_id, " steps_collected ", steps_collected,   flush=True)
        episode_rewards.append(np.sum(ep_rwds))
        # time.sleep(0.001)

def collect_expert_data(
                        expert,
                        num_steps_per_iter,
                        filename,
                        **env_kwargs,
                        ):
    
    with h5py.File(filename, 'w') as hf:
            pass  # This just opens and immediately closes the file, effectively clearing it

    exp_rwd_iter = []
    exp_obs = []
    exp_acts = []
    exp_done = []
    torch.set_num_threads(1)
    # Create the shared Manager object
    manager = torch.multiprocessing.Manager()
    # env_value = manager.Value(type(None), None)


    # Create a lock for workers
    lock = mp.Lock()
    
    processes_launched = multiprocessing.Event()

    # Initialize a queue to store expert data
    exp_data_queue = multiprocessing.Queue()

    # Initialize a list to store episode rewards
    episode_rewards = manager.list()

    # Determine the number of workers based on available CPU cores
    num_workers = max(multiprocessing.cpu_count()-5,1)

    # Calculate the number of steps per worker
    num_steps_per_worker = num_steps_per_iter // num_workers
    # num_steps_per_worker *=1.25 # allocate higher number of episodes than quota, so that free workers can do them w/o a blocking call

    # Create a list to store worker processes
    worker_processes = []

    # env_value.value = env
    # Launch worker processes for expert data collection

    for i in range(num_workers):
        
        worker_process = multiprocessing.Process(
                                                    target=worker, 
                                                    args=(
                                                            copy.deepcopy(expert), 
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

    # Collect expert data from the queue
    ep_collected = 0

    with h5py.File(filename, 'a') as hf:
        # Save the data list as an HDF5 file

        while len(exp_acts) < 0.9*num_steps_per_iter:
            ob, act, done = exp_data_queue.get()
            # print('act, done ', act, done, len(act))
            exp_obs.extend(ob)
            exp_acts.extend(act)
            exp_done.extend(done)
            ep_collected +=1
            episode_group = hf.create_group(f'episode_{ep_collected}')
            for i, (arr1, arr2, arr3) in enumerate(zip(ob, act, done)):
                episode_group.create_dataset(f'exp_obs{i}',  data=arr1)
                episode_group.create_dataset(f'exp_acts{i}', data=arr2)
                episode_group.create_dataset(f'exp_done{i}', data=arr3)
            pbar_outer.update(len(exp_acts) - pbar_outer.n)
    
    print(" joining worker processes ", [worker.pid for worker in worker_processes], flush=True)


    # Join worker processes to wait for their completion
    for worker_process in worker_processes:
        worker_process.terminate()

    print(" End of worker_process join")

    # Close and join the queue
    exp_data_queue.close()
    exp_data_queue.join_thread()
    pbar_outer.close()
    # Accumulate episode rewards where episodes are done
    for rwd in episode_rewards:
        exp_rwd_iter.append(rwd)

    
    exp_rwd_mean = np.mean(exp_rwd_iter)
    print(
        "Expert Reward Mean: {}".format(exp_rwd_mean)
    )

    exp_obs = np.array(exp_obs)
    exp_acts = np.array(exp_acts)
    exp_done = np.array(exp_done)



    return exp_obs, exp_acts, exp_done

def downsample_most_dominant_class(exp_obs, exp_acts, exp_dones):
    # Calculate the distribution of classes
    class_distribution = Counter(exp_acts)
    most_common_class, most_common_count = class_distribution.most_common(1)[0]
    second_most_common_class, second_most_common_count = class_distribution.most_common(2)[1]
    print("class_distribution ", class_distribution)
    print(
            "most_common_class ", most_common_class,
            "most_common_count ", most_common_count, 
            " second_most_common_class ", second_most_common_class, 
            " second_most_common_count ", second_most_common_count
         )

    # Determine whether downsampling is needed
    if most_common_count > 2 * second_most_common_count:
        # Calculate the desired number of samples for downsampling
        desired_samples = 2 * second_most_common_count

        # Perform downsampling on the most common class
        downsampled_exp_obs = []
        downsampled_exp_acts = []
        downsampled_exp_dones = []

        for ob, act, done in zip(exp_obs, exp_acts, exp_dones):
            # print(class_distribution, desired_samples)
            if act == most_common_class and class_distribution[act] > desired_samples:
                class_distribution[act] -= 1
                continue
                downsampled_exp_obs.append(ob)
                downsampled_exp_acts.append(second_most_common_class)
                downsampled_exp_dones.append(done)
            else:
                downsampled_exp_obs.append(ob)
                downsampled_exp_acts.append(act)
                downsampled_exp_dones.append(done)
        print(" downsampled_exp_acts count ", len(downsampled_exp_acts))
        return downsampled_exp_obs, downsampled_exp_acts, downsampled_exp_dones
    else:
        # No downsampling needed, return original data
        return exp_obs, exp_acts, exp_dones

