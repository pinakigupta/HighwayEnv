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

import functools
Observation = np.ndarray

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
            "lane_change_reward": 0.0,   # The reward received at each lane change action.
            "speed_reward_spd" : [5, 10, 15, 20, 25, 30],
            "speed_reward_rwd" : [-0.5 , -0.5, 0.0, 0.8, 1.0, 1.0],
            "travel_reward": 0.0,
            "imitation_reward": 0.0,
            "normalize_reward": False,
            "offroad_terminal": False
            # "reward_speed_range": [20, 30],
        })
        return config

    def __init__(self, config: dict = None, render_mode: Optional[str] = None, **kwargs) -> None:
        super().__init__(config, render_mode)
        for vehicle in self.road.vehicles:
                    if vehicle not in self.controlled_vehicles:
                        vehicle.check_collisions = False

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()
        self.ego_x0 = self.vehicle.position[0]
        self.cum_imitation_reward  = 0.0
        self.step({'long':'IDLE', 'lat':'STRAIGHT'})

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        if self.config["lanes_count"] == 'random':
            self.config["lanes_count"] = int(self.np_random.uniform(low=self.config["min_lanes_count"], high=self.config["max_lanes_count"]))
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"], speed_limit=30),
                         np_random=self.np_random, record_history=self.config["show_trajectories"], **self.config)

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        if self.config["vehicles_count"] == 'random':
            self.config["vehicles_count"] = self.np_random.integers(0, self.config['max_vehicles_count'])
        if self.config["vehicles_density"] == 'random':
            self.config["vehicles_density"] = self.np_random.uniform(low=0.5, high=10)

        other_per_controlled = near_split(self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"])

        self.controlled_vehicles = []
        speed = self.np_random.uniform(low=0, high=30)
        for others in other_per_controlled:
            vehicle = Vehicle.create_random(
                self.road,
                speed=speed,
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"],
                x0=100,
                action_type = self.action_type,
                **{
                    **self.config,
                    'LENGTH':self.config['EGO_LENGTH'],
                    'WIDTH':self.config['EGO_WIDTH']
                  }
            )
            vehicle = self.action_type.vehicle_class(
                                                        road = self.road, 
                                                        position = vehicle.position, 
                                                        heading = vehicle.heading, 
                                                        speed = vehicle.speed,
                                                        target_speed = 30, 
                                                        action_type = vehicle.action_type,
                                                        **{
                                                            **self.config,
                                                            'LENGTH':self.config['EGO_LENGTH'],
                                                            'WIDTH':self.config['EGO_WIDTH']
                                                          }                                                    
                                                      
                                                    )
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

            for _ in range(others):
                vehicle = other_vehicles_type.create_random(
                                                            self.road, 
                                                            spacing= 1 / self.config["vehicles_density"],
                                                            action_type = self.action_type,
                                                            length_noise = functools.partial(np.random.normal, loc=0, scale=0.25),
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
        # scaled_speed = utils.lmap(forward_speed, self.config["reward_speed_range"], [0, 1])
        travel_reward = 0
        if self._is_truncated():
            avg_speed = self.ego_travel/self.time
            travel_reward = np.clip(np.interp(avg_speed, self.config['speed_reward_spd'], self.config['speed_reward_rwd']),0,1)
            # travel_reward = 1.0 #self.ego_travel/1200.0
            # print("avg_speed ", avg_speed, " cum_imitation_reward  ", self.cum_imitation_reward )
    
        expert_action = self.vehicle._discrete_action
        imitation_reward = 0
        if action:
            for key, act, expert_act in zip(action.keys(), action.values(), expert_action.values()):
                try:
                    act = self.action_type.actions_indexes[key][act]
                    expert_act = self.action_type.actions_indexes[key][expert_act]
                    imitation_reward += 0.1*abs(act-expert_act)
                except Exception as e:
                    print(e)
            if imitation_reward == 0:
                imitation_reward -= 0.2
        self.cum_imitation_reward += imitation_reward
        # print(f'imitation_reward is {imitation_reward}')
        
        return {
            "collision_reward": float(self.vehicle.crashed),
            "right_lane_reward": 0 , #lane / max(len(neighbours) - 1, 1),
            "high_speed_reward": 0 , #np.clip(scaled_speed, 0, 1),
            # "on_road_reward": float(self.vehicle.on_road),
            "lane_change_reward": action['lat'] in [0, 2] if action else 0,
            "travel_reward": travel_reward,
            "imitation_reward": imitation_reward,
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
            if s < 0 : # or timegap < self.config['headway_timegap']:
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
    
    # def get_ep_reward(self):
    #     reward = self.ep_rward
    #     return reward
    
    def discrete_action(self, action):
        return action


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
