from copy import deepcopy as dcp
import torch.nn as nn
from torch import multiprocessing
import os, shutil
os.environ["HDF5_USE_THREADING"] = "true"
import h5py
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



# Custom feature extractor for the action space
class ActionFeatureExtractor(nn.Module):
    def __init__(self, action_space, dropout_factor=0.75, feature_size = 8, **kwargs):
        super(ActionFeatureExtractor, self).__init__()
        if isinstance(action_space, spaces.Discrete) or isinstance(action_space, gyms.spaces.discrete.Discrete):
            self.embeddings = nn.Embedding(action_space.n, feature_size)
            self.normalize = nn.BatchNorm1d(feature_size)
        elif isinstance(action_space, spaces.Box):
            self.embeddings = nn.Identity(action_space.shape[0])
            self.normalize = nn.BatchNorm1d(action_space.shape[0])
        self.action_space = action_space
        self.dropout_factor = dropout_factor
        # self.dropout = nn.functional.dropout(p=dropout_factor, training=True)
        self.activation = nn.Tanh()
        self.training = None
        if 'training' in kwargs:
            self.training = kwargs['training']

    def forward(self, action):
        if isinstance(self.action_space, spaces.Discrete) or isinstance(self.action_space, gyms.spaces.discrete.Discrete):
            # Add an extra dimension for the batch
            action = action.unsqueeze(1)
            embedded = self.embeddings(action.long()).squeeze(dim=1)
            normalized = self.normalize(embedded)
            activated = self.activation(normalized)
        elif isinstance(self.action_space, spaces.Box):
            # For Box action space, use an identity layer directly
            embedded = self.embeddings(action)
            normalized = self.normalize(embedded)
            activated = self.activation(normalized)
        if self.training:    
            return nn.functional.dropout(activated, p=self.dropout_factor, training=self.training)
        return  nn.functional.dropout(activated, p=self.dropout_factor)

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
    


class CombinedFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self,  observation_space: gym.spaces.Dict, **kwargs):
        features_dim = kwargs["attention_layer_kwargs"]["feature_size"]+kwargs["action_extractor_kwargs"]["feature_size"] 
        super().__init__(observation_space, features_dim = features_dim)
        self.obs_extractor = CustomExtractor(kwargs["action_extractor_kwargs"]['obs_space'], **kwargs)
        self.action_extractor = ActionFeatureExtractor(
                                                        action_space =    kwargs["action_extractor_kwargs"]['act_space'],
                                                        dropout_factor =  kwargs["action_extractor_kwargs"]['dropout_factor'],
                                                        feature_size =    kwargs["action_extractor_kwargs"]['feature_size']
                                                      )
        self.kwargs = kwargs
        linear =  nn.Linear(features_dim, features_dim)
        self.fc_layer = nn.Sequential(linear,  nn.ReLU(), linear, nn.ReLU())

    def forward(self, observations):
        # De-construct obs and act from observations using the original shapes
        obs_shape = self.kwargs["action_extractor_kwargs"]["obs_space"].shape
        action_shape = self.kwargs["action_extractor_kwargs"]["act_space"].shape

        # Separate obs and act while keeping the batch dimension
        obs = observations[:, :np.prod(obs_shape)].reshape(observations.shape[0], *obs_shape)
        act = observations[:, np.prod(obs_shape):].reshape(observations.shape[0], *action_shape)

        obs_features = self.obs_extractor(obs)
        action_features = self.action_extractor(act)

        combined_features = torch.cat([obs_features, action_features], dim=-1)
        return self.fc_layer(combined_features)



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


class CustomMLPPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, net_arch, features_extractor, **kwargs):
        super(CustomMLPPolicy, self).__init__(observation_space, action_space, lr_schedule, net_arch, features_extractor, **kwargs)

    def forward(self, obs, deterministic=False):
        # Custom logic for the forward pass
        # You can modify this as needed to define your policy's architecture
        features = self.extract_features(obs)
        action_mean = self.mlp_extractor(features)
        action_std = th.ones_like(action_mean)  # You can customize this for your action space
        value = self.value_net(features)
        
        if not deterministic:
            action = self.dist.sample()
        else:
            action = action_mean

        return action, value

class CustomMLPFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=64):
        super(CustomMLPFeaturesExtractor, self).__init__(observation_space, features_dim)
        # Define the architecture for feature extraction here
        self.features_extractor = nn.Sequential(
            nn.Linear(observation_space.shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        return self.features_extractor(observations)