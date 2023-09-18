from enum import Enum

env_kwargs = {
    'id': 'highway-v0',
    'render_mode': 'rgb_array',
    'expert': 'MDPVehicle',
    'config': {
        'deploy': False,
        "EGO_LENGTH": 5,
        "EGO_WIDTH": 2,
        'simulation_frequency': 10,
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
                "sin_h"
            ],
            "absolute": False,
            "relative_features": ['x']
        },
        # "observation": {
        #     "type": "GrayscaleObservation",
        #     "observation_shape": (128, 64),
        #     "stack_size": 4,
        #     "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
        #     "scaling": .75,
        # },
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 10,
            "features": [
                "presence",
                "x",
                "y",
                "vx",
                "vy",
                "cos_h",
                "sin_h"
            ],
            "absolute": False,
            "relative_features": ['x']
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

train = TrainEnum.BC
zip_filename = 'expert_data_trial.zip'