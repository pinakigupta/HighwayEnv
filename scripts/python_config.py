from enum import Enum
import numpy as np
import functools

env_kwargs = {
    'id': 'highway-v0',
    'render_mode': 'rgb_array',
    'expert': 'MDPVehicle',
    'config': {
        'deploy': False,
         **{
                "EGO_LENGTH": 'random',
                "EGO_WIDTH": 'random',
                "LENGTH": 'random',
                'WIDTH': 'random',
                "min_length": 4,
                "max_length": 10,
                "min_width": 2,
                "max_width": 3.5,
            },
        'position_noise': functools.partial(np.random.normal, loc=0, scale=0.25),
        'simulation_frequency': 10,
        "min_lanes_count": 2,
        "max_lanes_count": 7,
        "lanes_count": 'random',
        "other_vehicles_type": 'highway_env.vehicle.behavior.IDMVehicle',
        "vehicles_count": 'random',
        "max_vehicles_count": 10,
        'politeness': 0,
        'headway_timegap': 1.0,
        "action": {
                "type": "DiscreteMetaAction",
            },
        "offscreen_rendering": True,
        "KinematicObservation": {
            "type": "Kinematics",
            "vehicles_count": 10,
            "features": [
                "presence",
                "x",
                "y",
                "vx",
                "vy",
                "cos_h",
                "sin_h",
                'L',
                'W',
                'lane' 
            ],
            "absolute": False,
            "relative_features": ['x']
        },
        "GrayscaleObservation": {
            "type": "GrayscaleObservation",
            "observation_shape": (128, 64),
            "stack_size": 4,
            "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
            "scaling": .75,
        },
        "policy_frequency": 2,
        "duration": 40,
        "screen_width": 960,
        "screen_height": 180,
        "screen_text": False,
        "real_time_rendering": False
    }
}

sweep_config = {
    "method": "grid",
    "metric": {
        "name": "episode_reward",
        "goal": "maximize"
    },
    "parameters": {
        "duration": {
            "values": [40]  # Values for the "duration" field to be sappend_key_to_dict_of_dict(env_kwargs,'config','vehicles_count',150)wept
        },
        "gae_gamma": {
            "values": [0.995]  # Values for the "duration" field to be swept
        }, 
        "discrm_lr": {
            "values": [0.001]  # Values for the "duration" field to be swept
        },              
        "batch_size": {
            "values": [ 64 ]  # Values for the "duration" field to be swept
        }, 
        "num_epochs": {
            "values": [1]  # Values for the "duration" field to be swept
        },    
    }
}

class TrainEnum(Enum):
    RLTRAIN = 0
    RLDEPLOY = 1
    IRLTRAIN = 2
    IRLDEPLOY = 3
    EXPERT_DATA_COLLECTION =4
    BC = 5
    BCDEPLOY = 6
    ANALYSIS = 7
    VALIDATION = 8

project_names= \
    [
        f"RL",                       # RLTRAIN = 0
        f"RL",                       # RLDEPLOY = 1
        f"random_env_gail_1",        # IRLTRAIN = 2
        f"random_env_gail_1",        # IRLDEPLOY = 3
        f"BC" ,                      # EXPERT_DATA_COLLECTION =4
        f"BC" ,                      # BC = 5
        f"BC" ,                      # BCDEPLOY = 6
        f'None',                     # ANALYSIS = 7
        f'BC'                        # VALIDATION = 8
    ]

train = TrainEnum.BC
zip_filename = 'temp_1.zip'
# env_kwargs['config']['observation'] = env_kwargs['config']['GrayscaleObservation'] 
env_kwargs['config']['observation'] = env_kwargs['config']['KinematicObservation'] 

attention_network_kwargs = dict(
    # in_size=5*15,
    embedding_layer_kwargs={
                                "in_size": len(env_kwargs['config']['KinematicObservation']['features']), 
                                "layer_sizes": [64, 64], 
                                "reshape": False,
                                "activation": 'RELU'
                            },
    attention_layer_kwargs={
                                "feature_size": 64, 
                                "heads": 2, 
                                # "dropout_factor" :0.2
                           },
    # num_layers = 3,
)
label_weights = np.array([3, 1])

