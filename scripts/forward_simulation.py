import torch
from torch import multiprocessing
import gymnasium as gym
from highway_env.utils import print_overwrite
import concurrent.futures
import statistics
import numpy as np
from utils import record_videos
# ==================================
#     Environment configuration
# ==================================

total_count_lock = multiprocessing.Lock()
total_count = multiprocessing.Value("i", 0)

class DictToMultiDiscreteWrapper(gym.Wrapper):
    def __init__(self, env, key_order=None, **kwargs):
        super(DictToMultiDiscreteWrapper, self).__init__(env, **kwargs)
        
        if isinstance(self.env.action_space, gym.spaces.Dict):
            # Assuming that the 'action_space' of the original environment is a Dict
            self.key_order = key_order or list(self.env.action_space.spaces.keys())
            self.ndim = [self.env.action_space[key].n for key in self.key_order]
            self.action_space = gym.spaces.MultiDiscrete(self.ndim)
            # self.multi_discrete_action = self.action_space.sample()
            # self.action_space = self.convert_to_multi_discrete(self.env.action_space)
        else:
            self.key_order = None  # No key order needed for non-dict action spaces
            self.action_space = self.env.action_space

    def step(self, action):
        # If the environment uses Dict action space, convert the MultiDiscrete action to a Dict action
        if isinstance(self.env.action_space, gym.spaces.Dict):
            dict_action = self.convert_to_dict(action)
        else:
            dict_action = action
        return self.env.step(dict_action)
    
    def discrete_action(self, discrete_action):
        return self.convert_to_multi_discrete(self.env.discrete_action(discrete_action))

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def convert_to_multi_discrete(self, dict_action):
        # Convert a Dict action space to a MultiDiscrete action space
        action_list = [dict_action[key] for key in self.key_order]
        # self.action_space.sample()[:] = action_list
        return action_list

    def convert_to_dict(self, multi_discrete_action):
        # Convert a MultiDiscrete action to a Dict action
        if isinstance(multi_discrete_action, dict):
            return multi_discrete_action
        dict_action = {}
        for i, key in enumerate(self.key_order):
            discrete_action = multi_discrete_action[i] # This is numeric at this point
            dict_action[key] =  self.env.action_type.actions[key][discrete_action]
        return dict_action

class MultiDiscreteToSingleDiscreteWrapper(gym.Wrapper):
    def __init__(self, env, **kwargs):
        super(MultiDiscreteToSingleDiscreteWrapper, self).__init__(env, **kwargs)
        self.nvec = self.env.action_space.nvec
        self.action_weights = np.copy(self.nvec) # Action weights for mapping multi to single
        self.action_weights[-1] = 1
        self.inv_action_weights = 1/self.action_weights
        if isinstance(self.env.action_space, gym.spaces.MultiDiscrete):            
            self.action_space = gym.spaces.Discrete(np.prod(self.nvec))
        else:
            self.action_space = self.env.action_space

    def step(self, action):
        # Convert a MultiDiscrete action to a single action using action_weights
        if isinstance(self.env.action_space, gym.spaces.MultiDiscrete):
            multi_action = self.convert_to_multi_discrete(action)
        else:
            # Pass the action through as is
            multi_action = action
        return self.env.step(multi_action)
    
    def discrete_action(self, discrete_action):
        return self.convert_to_single_discrete(self.env.discrete_action(discrete_action))
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
    
    def convert_to_multi_discrete(self, single_action:gym.spaces)->gym.spaces:
        if  isinstance(self.env.action_space, gym.spaces.MultiDiscrete):
            # If the action space is not MultiDiscrete, return the input as is
            return single_action # Already multi 

        return single_action @ self.inv_action_weights
    
    def convert_to_single_discrete(self, multi_action:gym.spaces)->gym.spaces:
        return multi_action @ self.action_weights
        
def make_configure_env(**kwargs):
    env = gym.make(
                    # kwargs["id"], 
                    # render_mode=kwargs["render_mode"], 
                    # config=kwargs["config"],
                    **kwargs
                  )
    # env.configure(kwargs["config"])
    env.reset()
    custom_key_order = ['long','lat']
    env = DictToMultiDiscreteWrapper(env,key_order=custom_key_order)
    env = MultiDiscreteToSingleDiscreteWrapper(env)
    return env


def append_key_to_dict_of_dict(kwargs, outer_key, inner_key, value):
    kwargs[outer_key] = {**kwargs.get(outer_key, {}), inner_key: value}


def worker_rollout(worker_id, agent, render_mode, env_kwargs, gamma = 1.0, num_rollouts=50, num_workers =4):
    global total_count
    rollouts_per_worker = num_rollouts // num_workers
    extra_rollouts = num_rollouts % num_workers
    # print("rollouts_per_worker ", rollouts_per_worker, "extra_rollouts ", extra_rollouts)

    total_rewards = []

    if worker_id != 0:
        env_kwargs.update({'render_mode': 'rgb_array'})
    else:
        env_kwargs.update({'render_mode': render_mode})
        append_key_to_dict_of_dict(env_kwargs,'config','real_time_rendering',True)
    
    env = make_configure_env(**env_kwargs)
    record_videos(env=env, name_prefix = 'GAIL', video_folder='videos/GAIL')
    for _ in range(rollouts_per_worker):
        obs, info = env.reset()
        done = truncated = False
        cumulative_reward = 0
        while not (done or truncated):
            with torch.no_grad():
                try:  
                    # print(type(agent))  
                    action = agent.act(obs.flatten())
                except:
                    try:
                        action = agent.predict(obs)
                        action = action[0]
                    except:
                        action = agent.pi(obs)
            obs, reward, done, truncated, info = env.step(action)
            cumulative_reward += gamma * reward
            # xs = [(v.position[0],v.speed) for v in env.road.vehicles]
            # print(" ego x " , env.vehicle.position[0], "xs ", xs)
            # print("speed: ",env.vehicle.speed," ,reward: ", reward, " ,cumulative_reward: ",cumulative_reward)
            # env.render(render_mode=render_mode)
        total_rewards.append(cumulative_reward)
        count_length = len("Total Episodes Count: ")  # Length of the text before the count
        with total_count_lock:
            total_count.value += 1
            count_text = f"Total Episodes Count: {total_count.value}"
            print_overwrite(count_text, count_length)
        # print(worker_id," : ",len(total_rewards),"--------------------------------------------------------------------------------------")
    return total_rewards

def simulate_with_model( agent, env_kwargs, render_mode, gamma = 1.0, num_rollouts=50, num_workers = 4):
    # progress_bar = tqdm(total=num_rollouts , desc="Episodes", unit="episodes")
    all_rewards = []
    work_queue = list(range(num_rollouts))  # Create a work queue with rollout IDs
    completed_rollouts = 0  # Counter for completed rollouts
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_rollout = {executor.submit(
                                                    worker_rollout, 
                                                    worker_id, 
                                                    agent, 
                                                    render_mode=render_mode,
                                                    env_kwargs=env_kwargs, 
                                                    num_workers=num_workers, 
                                                    num_rollouts=num_rollouts
                                                ):
                                worker_id for worker_id in range(num_workers)}

            while completed_rollouts < num_rollouts:
                # Wait for the next completed rollout and get its result
                for future in concurrent.futures.as_completed(future_to_rollout):
                    worker_id = future_to_rollout[future]
                    try:
                        rewards = future.result()
                        all_rewards.extend(rewards)
                        completed_rollouts += 1

                        if completed_rollouts > num_rollouts:
                            break

                        if work_queue:
                            next_rollout = work_queue.pop(0)
                            future_to_rollout[executor.submit(worker_rollout, worker_id, agent, render_mode=render_mode,
                                                            env_kwargs=env_kwargs, num_workers=num_workers,
                                                            num_rollouts=num_rollouts)] = worker_id
                    except concurrent.futures.process.BrokenProcessPool as e:
                        print("BrokenProcessPool Exception:", e)

    mean_rewards = statistics.mean(all_rewards)
    return mean_rewards
