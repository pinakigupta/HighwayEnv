from typing import Dict, Text

import numpy as np
from typing import Optional
from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle

Observation = np.ndarray
speed_reward_spd = [5, 10, 15, 20, 25, 30]
speed_reward_rwd = [-0.5 , -0.5, 0.0, 0.8, 1.0, 1.0]

class HighwayEnv(AbstractEnv):
    """
    A highway driving environment.

    The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed,
    staying on the rightmost lanes and avoiding collisions.
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics",
                "relative_features": ['x'],
            },
            "action": {
                "type": "DiscreteMetaAction",
            },
            "lanes_count": 4,
            "vehicles_count": 50,
            "controlled_vehicles": 1,
            "initial_lane_id": None,
            "duration": 40,  # [s]
            "ego_spacing": 2,
            "vehicles_density": 1,
            "collision_reward": -1,    # The reward received when colliding with a vehicle.
            "right_lane_reward": 0.0,  # The reward received when driving on the right-most lanes, linearly mapped to
                                       # zero for other lanes.
            "high_speed_reward": 0.0,  # The reward received when driving at full speed, linearly mapped to zero for
                                       # lower speeds according to config["reward_speed_range"].
            "lane_change_reward": -0.05,   # The reward received at each lane change action.
            "reward_speed_range": [20, 30],
            "travel_reward": 2,
            "normalize_reward": False,
            "offroad_terminal": False
        })
        return config

    def __init__(self, config: dict = None, render_mode: Optional[str] = None, **kwargs) -> None:
        super().__init__(config, render_mode)
        self.reward_oracle = kwargs["reward_oracle"] if "reward_oracle" in kwargs else None
        for vehicle in self.road.vehicles:
                    if vehicle not in self.controlled_vehicles:
                        vehicle.check_collisions = False

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()
        self.ego_x0 = self.vehicle.position[0]
        self.step(4)

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"], speed_limit=30),
                         np_random=self.np_random, record_history=self.config["show_trajectories"])

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        if self.config["vehicles_count"] == 'random':
            self.config["vehicles_count"] = self.np_random.integers(0, self.config['max_vehicles_count'])
        if self.config["vehicles_density"] == 'random':
            self.config["vehicles_density"] = self.np_random.uniform(low=0.5, high=10)
        other_per_controlled = near_split(self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"])

        self.controlled_vehicles = []
        speed = self.np_random.uniform(low=20, high=30)
        for others in other_per_controlled:
            vehicle = Vehicle.create_random(
                self.road,
                speed=speed,
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"],
                x0=100,
                **self.config
            )
            vehicle = self.action_type.vehicle_class(self.road, vehicle.position, vehicle.heading, vehicle.speed, **self.config)
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

            for _ in range(others):
                vehicle = other_vehicles_type.create_random(
                                                            self.road, 
                                                            spacing= 1 / self.config["vehicles_density"],
                                                            **self.config
                                                           )
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)

    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        rewards = self._rewards(action)
        weighed_reward = [self.config.get(name, 0) * reward for name, reward in rewards.items()]
        reward = sum(weighed_reward)
        obs = self.observation_type.observe()
        if hasattr(self, 'reward_oracle') and self.reward_oracle is not None:
            import torch
            from torch import FloatTensor, squeeze, zeros
            obs = FloatTensor(obs)
            obs = obs.view(1, -1) 
            action = torch.tensor(action, dtype=torch.int64) 
            action = action.view(1)
            # print(self.reward_oracle, "obs.shape ",obs.shape, " action.shape ", action.shape)
            expert_reward = self.reward_oracle(obs , action )
            expert_reward = expert_reward[0].item()
            print(" expert_reward ", expert_reward)
            # if expert_reward < 0.2:
            #     reward += 0.1
        # if self.config["normalize_reward"]:
        #     reward = utils.lmap(reward,
        #                         [
        #                             self.config["collision_reward"],
        #                             (self.config["high_speed_reward"] + self.config["right_lane_reward"] + self.config["travel_reward"])/\
        #                             (self.config["duration"]*self.config["policy_frequency"])
        #                          ],
        #                         [0, 1]
        #                         )
        # reward *= rewards['on_road_reward']
        
        return reward

    def _rewards(self, action: Action) -> Dict[Text, float]:
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index[2]
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(forward_speed, self.config["reward_speed_range"], [0, 1])
        travel_reward = 0
        if self._is_truncated():
            avg_speed = self.ego_travel/self.time
            travel_reward = np.clip(np.interp(avg_speed, speed_reward_spd, speed_reward_rwd),0,1)
            # print("travel_reward ", travel_reward)
    

        return {
            "collision_reward": float(self.vehicle.crashed),
            "right_lane_reward": 0 , #lane / max(len(neighbours) - 1, 1),
            "high_speed_reward": 0 , #np.clip(scaled_speed, 0, 1),
            # "on_road_reward": float(self.vehicle.on_road),
            "lane_change_reward": action in [0, 2] ,
            "travel_reward": travel_reward,
            #np.clip(float(self._is_truncated()), 0, 1) ,
        }

    def _is_terminated(self) -> bool:
        """The episode is over if the ego vehicle crashed."""
        terminated = (self.vehicle.crashed or
                self.config["offroad_terminal"] and not self.vehicle.on_road)
        front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self.vehicle, self.vehicle.lane_index)
        if front_vehicle and (not self.config['deploy']):
            s = self.vehicle.lane_distance_to(front_vehicle)
            timegap = s/max(self.vehicle.speed,1.0)
            if s < self.vehicle.LENGTH/2 or timegap < self.config['headway_timegap']:
                terminated = True
                self.vehicle.crashed = True
        return terminated

    def _is_truncated(self) -> bool:
        """The episode is truncated if the time limit is reached."""
        truncated =  self.time >= self.config["duration"]
        return truncated
    
    def ego_avg_speed(self):
        return self.vehicle.speed
    
    def ep_duration(self):
        return self.time
    
    def travel_reward(self):
        if self._is_truncated() or self.done:
            travel = self.vehicle.position[0] - self.ego_x0
            print('travel ', travel)
        return 0
    
    def step(self, action: Action):
        self.ego_travel = self.ego_travel_eval() # sure a step delayed
        return super().step(action)
    
    def ego_travel_eval(self):
        return self.vehicle.position[0] - self.ego_x0
    
    def set_config(self, config):
        self.config = config

    def get_config(self):
        return self.config
    
    def get_ep_reward(self):
        reward = self.ep_rward
        self.ep_rward = 0
        return reward


class HighwayEnvFast(HighwayEnv):
    """
    A variant of highway-v0 with faster execution:
        - lower simulation frequency
        - fewer vehicles in the scene (and fewer lanes, shorter episode duration)
        - only check collision of controlled vehicles with others
    """
    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({
            "simulation_frequency": 5,
            "lanes_count": 3,
            "vehicles_count": 20,
            "duration": 30,  # [s]
            "ego_spacing": 1.5,
        })
        return cfg

    def _create_vehicles(self) -> None:
        super()._create_vehicles()
        # Disable collision check for uncontrolled vehicles
        for vehicle in self.road.vehicles:
            if vehicle not in self.controlled_vehicles:
                vehicle.check_collisions = False
