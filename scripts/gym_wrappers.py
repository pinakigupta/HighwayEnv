import torch
from torch import multiprocessing
import gymnasium as gym
from highway_env.utils import print_overwrite
import concurrent.futures
import statistics
import numpy as np
from utils import record_videos
import random
import torch.nn.functional as F

class DictToMultiDiscreteWrapper(gym.Wrapper):
    def __init__(self, env, key_order=None, **kwargs):
        super(DictToMultiDiscreteWrapper, self).__init__(env, **kwargs)
        
        if isinstance(self.env.action_space, gym.spaces.Dict):
            # Assuming that the 'action_space' of the original environment is a Dict
            self.key_order = key_order or list(self.env.action_space.spaces.keys())
            self.ndim = [self.env.action_space[key].n for key in self.key_order]
            self.action_space = gym.spaces.MultiDiscrete(self.ndim)
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
        action_list = [self.env.action_type.actions_indexes[key][dict_action[key]] for key in self.key_order] 
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
    
    
class ObsToDictObsWrapper(gym.Wrapper):
    def __init__(self, env,  **kwargs):
        super(ObsToDictObsWrapper, self).__init__(env, **kwargs)
        self.observation_space = gym.spaces.Dict({"obs": self.env.observation_space, "action": self.env.action_space})
        self.action_space = self.env.action_space

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        # Modify the observation to match the custom observation space
        custom_obs = {"obs": obs, "action": np.zeros(self.custom_action_space.shape)}
        return custom_obs

    def step(self, action):
        # Modify the action to match the custom action space
        obs, reward, done, info = self.env.step(action)
        # Modify the observation to match the custom observation space
        custom_obs = {"obs": obs, "action": action}
        return custom_obs, reward, done, info


class SquashObservationsWrapper(gym.Wrapper):
    def __init__(self, env, policy=None, expert_policy=None, **kwargs):
        super(SquashObservationsWrapper, self).__init__(env, **kwargs)
        self.action_space = self.env.action_space
        # Calculate obs_dim based on the shape of the observation space
        self.obs_dim = int(np.prod(env.observation_space.shape))
        self.expert_divergence = 0.0
        self.shuffled_indices = []
        if not self.config['deploy'] and 'obj_obs_random_shuffle_probability' in self.config and random.random() < self.config['obj_obs_random_shuffle_probability']:
            vehicles_count =  self.observation_type.observe().shape[0]
            shuffled_indices = list(range(1,vehicles_count)) 
            random.shuffle(shuffled_indices)
            self.shuffled_indices = shuffled_indices

        if isinstance(env.action_space, gym.spaces.Discrete):
            self.action_dim = env.action_space.n  # For Discrete action space, action_dim is the number of discrete actions
        else:
            self.action_dim = env.action_space.shape[0]  # For other action spaces, action_dim is the shape


        # Define the combined observation space
        low = np.concatenate([env.observation_space.low.flatten(), [0]])
        high = np.concatenate([env.observation_space.high.flatten(), [self.action_dim - 1]])  # Assuming discrete actions are in the range [0, action_dim-1]
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        custom_obs = np.concatenate([obs.flatten(), [0]])
        self.custom_reward = 0
        return custom_obs, info

    
    def update_expert_divergence(self, kl_divergence):
        self.expert_divergence = kl_divergence

    def get_expert_divergence(self):
        return self.expert_divergence
    
    def shuffle_obs(self, obs):
        if not self.shuffled_indices:
            return obs
        shuffled_obs = obs
        for idx, obs_idx in enumerate(self.shuffled_indices):
            shuffled_obs[idx, :] = obs[obs_idx, :]
        return shuffled_obs
        
    
    def get_obs(self):
        obs = self.observation_type.observe()
        obs = self.shuffle_obs(obs)
        custom_obs = np.concatenate([obs.flatten(), [0]])
        custom_obs[9::10] = 0 
        return custom_obs
    
    def step(self, action):
        obs, reward, done, truncated , info = self.env.step(action)
        obs = self.shuffle_obs(obs)
        custom_obs = np.concatenate([obs.flatten(), [0]]) 
        custom_obs[9::10] = 0 # hardcoding lane ids out 
        # if self.policy and self.expert_policy:
        #     with torch.no_grad():
        # self.custom_obs = custom_obs
        custom_reward = reward + self.config['kl_divergence_reward']*self.expert_divergence
        self.custom_reward = custom_reward
        return custom_obs, custom_reward, done, truncated, info
    
    def get_reward(self):
        return self.custom_reward


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
    
    def convert_to_multi_discrete(self, action:gym.spaces)->gym.spaces:
        if  isinstance(action, gym.spaces.Discrete) or isinstance(action, np.ndarray) or isinstance(action, np.int64) or isinstance(action, np.int32):
            # Hardcoding this for two variables for now
            rem = action%self.action_weights[0]
            mod = (action-rem)/self.action_weights[0]
            return np.array([mod, rem])
        
        return action # Already multi 

        
    
    def convert_to_single_discrete(self, multi_action:gym.spaces)->gym.spaces:
        return multi_action @ self.action_weights


def make_configure_env(policy=None, expert_policy=None, **kwargs):
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
    env = SquashObservationsWrapper(env, policy=policy, expert_policy=expert_policy)
    return env