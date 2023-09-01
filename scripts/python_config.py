
env_kwargs = {
    'id': 'highway-v0',
    'render_mode': 'rgb_array',
    'expert': 'MDPVehicle',
    'config': {
        'simulation_frequency': 10,
        "lanes_count": 4,
        "other_vehicles_type": 'highway_env.vehicle.behavior.IDMVehicle',
        "vehicles_count": 'random',
        'politeness': 0,
        "action": {
                "type": "DiscreteMetaAction",
            },
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
            "values": [40]  # Values for the "duration" field to be swept
        },
        "gae_gamma": {
            "values": [0.995]  # Values for the "duration" field to be swept
        }, 
        "discrm_lr": {
            "values": [0.001]  # Values for the "duration" field to be swept
        },              
        "batch_size": {
            "values": [ 128 ]  # Values for the "duration" field to be swept
        }, 
        "num_epochs": {
            "values": [10]  # Values for the "duration" field to be swept
        },    
    }
}
