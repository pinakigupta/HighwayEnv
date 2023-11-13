import statistics
import torch
import torch.nn.functional as F
import numpy as np
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
        if ep_rew_mean > 0.5:
            config = self.training_env.env_method("get_config")[0]
            config['duration'] += 5
            config['max_vehicles_count'] = int(1.05 *  config['max_vehicles_count'])
            self.training_env.env_method("set_config", config)
            print("Updated env duration to ", self.training_env.env_method("get_config")[0]['duration'])
        print("----------------------------------------------------------------------------------------")
        # Clear the episode_rewards list for the next epoch
        return True


class KLDivergenceCallback(BaseCallback):
    def __init__(self, expert_policy, kl_coefficient, verbose=0):
        super(KLDivergenceCallback, self).__init__(verbose)
        self.expert_policy = expert_policy
        self.kl_coefficient = kl_coefficient

    def _on_step(self) -> bool:
        # Access the PPO policy from the model

        # Access the observation without resetting the environmenting
        obs = self.training_env.env_method("get_obs")  # Replace with the correct way to get observations

        for env_idx in range(self.training_env.num_envs):
            with torch.no_grad():
                obs_with_batch = torch.Tensor(obs[env_idx]).to(self.model.policy.device).unsqueeze(0)
                action_distribution = self.model.policy.get_distribution(obs_with_batch)
                action_distribution = action_distribution.distribution.logits
                expert_action_distribution = self.expert_policy.get_distribution(obs_with_batch)
                expert_action_distribution = expert_action_distribution.distribution.logits
                action_probabilities = F.softmax(action_distribution, dim=1).cpu().numpy()
                expert_action_probabilities = F.softmax(expert_action_distribution, dim=1).cpu().numpy()
                kl_divergence = np.sum(action_probabilities * np.log(action_probabilities / expert_action_probabilities))
                # are_weights_same = all(torch.allclose(param1, param2) for param1, param2 in zip(self.model.policy.parameters(), self.expert_policy.parameters() ))
                # print(f'kl_divergence {kl_divergence}. are_weights_same { are_weights_same }.')
                self.training_env.env_method("update_expert_divergence", kl_divergence , indices=env_idx)

        return True