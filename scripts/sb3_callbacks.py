import statistics

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback


class CustomCheckpointCallback(BaseCallback):
    def __init__(self, save_freq, save_path):
        super(CustomCheckpointCallback, self).__init__()
        self.save_freq = save_freq
        self.save_path = save_path

    def _init_callback(self) -> None:
        if self.save_freq > 0:
            self.model.save(self.save_path)  # Save the initial model

    def _on_step(self) -> bool:
        # if self.n_calls % self.save_freq == 0:
        #     self.model.save(self.save_path)  # Save the model at specified intervals
        return True

    def _on_rollout_end(self) -> bool:
        # This method is called at the end of each episode
        ego_travel = statistics.mean(self.training_env.env_method("ego_travel_eval"))
        ep_duration = statistics.mean(self.training_env.env_method("ep_duration"))
        print("Mean speed for this episode:", ego_travel/ep_duration)
        print("Mean episode duration:", ep_duration)
        print("Mean episode ego travel:", ego_travel)
        return True

class CustomMetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CustomMetricsCallback, self).__init__(verbose)
        self.episode_lengths = []
        self.episode_rewards = []

    def _on_step(self) -> bool:
        # Retrieve the episode length and reward from the environment
        episode_info = self.locals.get("episode")
        if episode_info is not None:
            ep_len = episode_info["length"]
            ep_rew = episode_info["reward"]

            # Store the episode length and reward
            self.episode_lengths.append(ep_len)
            self.episode_rewards.append(ep_rew)

class CustomCurriculamCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CustomCurriculamCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Access the current episode reward from the environment
        return True

    def _on_rollout_end(self) -> bool:
        # Calculate the average episode reward after training ends
        ep_rew_mean = statistics.mean(self.training_env.env_method("get_ep_reward"))
        print("Mean episode ep_rew_mean:", ep_rew_mean)
        if ep_rew_mean > 0.1:
            config = self.training_env.env_method("get_config")[0]
            config['duration'] += 10
            self.training_env.env_method("set_config", config)
            print("Updated env duration to ", self.training_env.env_method("get_config")[0]['duration'])
        print("----------------------------------------------------------------------------------------")
        # Clear the episode_rewards list for the next epoch
        return True

