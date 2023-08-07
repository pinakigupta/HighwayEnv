
env_kwargs = {
    'id': 'highway-v0',
    'render_mode': 'rgb_array',
    'config': {
        "lanes_count": 4,
        "vehicles_count": 'random',
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
            "absolute": False
        },
        "policy_frequency": 2,
        "duration": 40,
        "screen_width": 960,
        "screen_height": 180,
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
            "values": [0.95, 0.99]  # Values for the "duration" field to be swept
        }, 
        "discrm_lr": {
            "values": [0.0005, 0.001, None]  # Values for the "duration" field to be swept
        },                  
    }
}