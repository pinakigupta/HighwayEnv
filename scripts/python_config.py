from enum import Enum
import numpy as np
import functools


env_kwargs = {
    'id': 'highway-v0',
    'render_mode': 'rgb_array',
    'expert': 'MDPVehicle',
    'config': {   #Vehicle configs
        'deploy': False,
        'obj_obs_random_shuffle_probability': 0.0,
         **{
                "EGO_LENGTH": 'random',
                "EGO_WIDTH": 'random',
                "LENGTH": 'random',
                'WIDTH': 'random',
                "min_length": 4,
                "max_length": 10,
                "min_width": 2,
                "max_width": 3.5,
                'politeness': 'random',
                "vehicles_count": 'random',
                "max_vehicles_count": 150,
                'headway_timegap': 1.0,
                "other_vehicles_type": 'highway_env.vehicle.behavior.IDMVehicle',
                'position_noise': functools.partial(np.random.normal, loc=0, scale=0.25),
                # 'length_noise': functools.partial(np.random.normal, loc=0, scale=0.25),
                'planning_heuristic': False,
            },
            **{
                "collision_reward": -1.0,    # The reward received when colliding with a vehicle.
                "right_lane_reward": 0.0,  # The reward received when driving on the right-most lanes, linearly mapped to
                                        # zero for other lanes.
                "high_speed_reward": 0.0,  # The reward received when driving at full speed, linearly mapped to zero for
                                        # lower speeds according to config["reward_speed_range"].
                "lane_change_reward": 0.0,   # The reward received at each lane change action.
                "speed_reward_spd" : [5, 10, 15, 20, 25, 30],
                "speed_reward_rwd" : [0.0 , 0.0, 0.5, 0.8, 1.0, 1.0],
                "travel_reward": 1.0,
                "imitation_reward": -0.05,
                "kl_divergence_reward": -0.05,
                "normalize_reward": False,
              },
        'simulation_frequency': 10,
            **{ # Lane configs
                "lanes_count": 'random',
                "min_lanes_count": 2,
                "max_lanes_count": 7,
             },
        "action": {
                "type": "DiscreteMetaAction",
            },
        "offscreen_rendering": True,
         ** { # Observation configs
                "KinematicObservation": {
                    "type": "Kinematics",
                    "vehicles_count": 10,
                    "features": [ "presence", "x", "y", "vx", "vy", "cos_h", "sin_h", 'L', 'W', 'lane'],
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
            },
          **{
                "screen_width": 960,
                "screen_height": 180,
                "screen_text": False,
                "real_time_rendering": False
              
            },
        "policy_frequency": 2,
        "duration": 40,
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
            "values": [ 1024 ]  # Values for the "duration" field to be swept
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
    BCVALIDATION = 8
    RLVALIDATION = 9

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
        f'BC',                       # BCVALIDATION = 8
        f'RL',                       # RLVALIDATION = 8
    ]

train = TrainEnum.RLTRAIN
# zip_filename = \
#                 [
#                     'temp_5.zip',
#                     'temp_7.zip',
#                     'temp_10.zip', 
#                     'temp_11.zip', 
#                     'temp_12.zip', 
#                     'temp_13.zip', 
#                     'temp_14.zip',
#                     'temp_15.zip',
#                     'temp_16.zip',
#                     'CL_temp_17.zip',                     
#                     'temp_18.zip',
#                     'temp_19.zip',
#                     'temp_20.zip',
#                     'temp_21.zip',
#                     'temp_22.zip',
#                 ]
zip_filename =    'temp_1.zip' 
# env_kwargs['config']['observation'] = env_kwargs['config']['GrayscaleObservation'] 
env_kwargs['config']['observation'] = env_kwargs['config']['KinematicObservation'] 

attention_network_kwargs = dict(
    # in_size=5*15,
    embedding_layer_kwargs={
                                "in_size": len(env_kwargs['config']['KinematicObservation']['features']), 
                                "layer_sizes": [128, 256, 256], 
                                "reshape": False,
                                "activation": 'RELU',
                                'dropout_factor': 0.2
                            },
    attention_layer_kwargs={
                                "feature_size": 256, 
                                "heads": 8, 
                                "dropout_factor" :0.25
                           },
    # num_layers = 3,
)

# label_weights = np.array([3, 1])

