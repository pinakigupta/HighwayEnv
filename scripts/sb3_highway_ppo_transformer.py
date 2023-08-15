import functools
import gymnasium as gym
import pygame
import seaborn as sns
import torch as th
from highway_env.utils import lmap, print_overwrite
from stable_baselines3 import PPO
from torch.distributions import Categorical
import torch
import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

import os, statistics
import multiprocessing
from enum import Enum
import json
import copy
import wandb
from datetime import datetime
from torch import FloatTensor
from torch import nn
import shutil
from models.nets import Expert
from models.gail import GAIL
from generate_expert_data import collect_expert_data, downsample_most_dominant_class
from sb3_callbacks import CustomCheckpointCallback, CustomMetricsCallback, CustomCurriculamCallback
from attention_network import EgoAttentionNetwork
from utilities import extract_expert_data, write_module_hierarchy_to_file, DefaultActorCriticPolicy
import warnings
import concurrent.futures
from stable_baselines3.common.evaluation import evaluate_policy
from imitation.algorithms import bc
from imitation.data import types
from python_config import sweep_config, env_kwargs
import matplotlib.pyplot as plt
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.ppo import MlpPolicy

from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.data import types

from ray import tune
from ray.tune.trainable import trainable

from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")




class TrainEnum(Enum):
    RLTRAIN = 0
    RLDEPLOY = 1
    IRLTRAIN = 2
    IRLDEPLOY = 3
    EXPERT_DATA_COLLECTION =4
    BC = 5
    BCDEPLOY = 6

train = TrainEnum.BCDEPLOY

def append_key_to_dict_of_dict(kwargs, outer_key, inner_key, value):
    kwargs[outer_key] = {**kwargs.get(outer_key, {}), inner_key: value}

attention_network_kwargs = dict(
    # in_size=5*15,
    embedding_layer_kwargs={"in_size": 7, "layer_sizes": [64, 64], "reshape": False},
    attention_layer_kwargs={"feature_size": 64, "heads": 2},
    # num_layers = 3,
)



class CustomExtractor(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, **kwargs):
        super().__init__(observation_space, features_dim=kwargs["attention_layer_kwargs"]["feature_size"])
        self.extractor = EgoAttentionNetwork(**kwargs)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.extractor(observations)

# ==================================
#     Environment configuration
# ==================================

def make_configure_env(**kwargs):
    env = gym.make(
                    # kwargs["id"], 
                    # render_mode=kwargs["render_mode"], 
                    # config=kwargs["config"],
                    **kwargs
                  )
    # env.configure(kwargs["config"])
    env.reset()
    return env





# ==================================
#        Display attention matrix
# ==================================

def display_vehicles_attention(agent_surface, sim_surface, env, fe, min_attention=0.01):
        v_attention = compute_vehicles_attention(env, fe)
        # print("v_attention ", v_attention)
        # Extract the subsurface of the larger rectangle
        attention_surface = pygame.Surface(sim_surface.get_size(), pygame.SRCALPHA)
        pygame.draw.circle(
                                        surface=attention_surface,
                                        color=pygame.Color("white"),
                                        center=sim_surface.vec2pix(env.vehicle.position),
                                        radius=20,
                                        width=2
                          )
        for head in range(list(v_attention.values())[0].shape[0]):
            
            for vehicle, attention in v_attention.items():
                if attention[head] < min_attention:
                    continue
                # if True: 
                #     print("attention[head] ", attention[head], "vehicle ", vehicle)
                width = attention[head] * 5
                desat = np.clip(lmap(attention[head], (0, 0.5), (0.7, 1)), 0.7, 1)
                colors = sns.color_palette("dark", desat=desat)
                color = np.array(colors[(2*head) % (len(colors) - 1)]) * 255
                color = (*color, np.clip(lmap(attention[head], (0, 0.5), (100, 200)), 100, 200))
                pygame.draw.line(attention_surface, color,
                                     sim_surface.vec2pix(env.vehicle.position),
                                     sim_surface.vec2pix(vehicle.position),
                                     max(sim_surface.pix(width), 1)
                                )
            # subsurface = attention_surface.subsurface(pygame.Rect(0, 0, 4800, 200))
            sim_surface.blit(attention_surface, (0, 0))

def compute_vehicles_attention(env,fe):
    obs = env.unwrapped.observation_type.observe()
    obs_t = torch.tensor(obs[None, ...], dtype=torch.float)
    attention = fe.extractor.get_attention_matrix(obs_t)
    attention = attention.squeeze(0).squeeze(1).detach().cpu().numpy()
    ego, others, mask = fe.extractor.split_input(obs_t)
    mask = mask.squeeze()
    v_attention = {}
    obs_type = env.observation_type
    if hasattr(obs_type, "agents_observation_types"):  # Handle multi-model observation
        obs_type = obs_type.agents_observation_types[0]
    for v_index in range(obs.shape[0]):
        if mask[v_index]:
            continue
        v_position = {}
        for feature in ["x", "y"]:
            v_feature = obs[v_index, obs_type.features.index(feature)]
            v_feature = lmap(v_feature, [-1, 1], obs_type.features_range[feature])
            v_position[feature] = v_feature
        v_position = np.array([v_position["x"], v_position["y"]])
        if not obs_type.absolute and v_index > 0:
            v_position += env.unwrapped.vehicle.position # This is ego
        vehicle = min(env.unwrapped.road.vehicles, key=lambda v: np.linalg.norm(v.position - v_position))
        v_attention[vehicle] = attention[:, v_index]
    return v_attention

# ==================================
#        Main script  20 
# ==================================

def retrieve_gail_agents(env, artifact_version="trained_model_directory:latest", project = None):
    state_dim = env.observation_space.high.shape[0]*env.observation_space.high.shape[1]
    action_dim = env.action_space.n
    observation_space = env.observation_space
    # Initialize wandb
    wandb.init(project=project, name="inference")
    # Access the run containing the logged artifact

    # Download the artifact
    artifact = wandb.use_artifact(artifact_version)
    artifact_dir = artifact.download()
    wandb.finish()

    # Load the model from the downloaded artifact
    optimal_gail_agent_path = os.path.join(artifact_dir, "optimal_gail_agent.pth")
    final_gail_agent_path = os.path.join(artifact_dir, "final_gail_agent.pth")

    with open("config.json") as f:
        config = copy.deepcopy(json.load(f))
    # Load the GAIL model
    final_gail_agent = GAIL(
                                state_dim, 
                                action_dim, 
                                discrete=True, 
                                device=torch.device("cpu"), 
                                **config, 
                            #  **policy_kwargs, 
                                observation_space= observation_space 
                                )
    optimal_gail_agent = copy.deepcopy(final_gail_agent)
    final_gail_agent.load_state_dict(torch.load(final_gail_agent_path))
    optimal_gail_agent.load_state_dict(torch.load(optimal_gail_agent_path))
    return optimal_gail_agent, final_gail_agent

total_count_lock = multiprocessing.Lock()
total_count = multiprocessing.Value("i", 0)



def worker_rollout(worker_id, agent, render_mode, env_kwargs, gamma = 1.0, num_rollouts=50, num_workers =4):
    global total_count
    rollouts_per_worker = num_rollouts // num_workers
    extra_rollouts = num_rollouts % num_workers
    # print("rollouts_per_worker ", rollouts_per_worker, "extra_rollouts ", extra_rollouts)

    total_rewards = []

    if worker_id != 0:
        env_kwargs.update({'render_mode': 'rgb_array'})
    else:
        env_kwargs.update({'render_mode': render_mode})
        append_key_to_dict_of_dict(env_kwargs,'config','real_time_rendering',True)
    
    env = make_configure_env(**env_kwargs)
    for _ in range(rollouts_per_worker):
        obs, info = env.reset()
        done = truncated = False
        cumulative_reward = 0
        while not (done or truncated):
            with torch.no_grad():
                try:    
                    action = agent.act(obs)
                except:
                    action = agent.predict(obs)
                    action = action[0]
            obs, reward, done, truncated, info = env.step(action)
            cumulative_reward += gamma * reward
            # xs = [(v.position[0],v.speed) for v in env.road.vehicles]
            # print(" ego x " , env.vehicle.position[0], "xs ", xs)
            # print("speed: ",env.vehicle.speed," ,reward: ", reward, " ,cumulative_reward: ",cumulative_reward)
            # env.render(render_mode=render_mode)
        total_rewards.append(cumulative_reward)
        count_length = len("Total Episodes Count: ")  # Length of the text before the count
        with total_count_lock:
            total_count.value += 1
            count_text = f"Total Episodes Count: {total_count.value}"
            print_overwrite(count_text, count_length)
        # print(worker_id," : ",len(total_rewards),"--------------------------------------------------------------------------------------")
    return total_rewards

def simulate_with_model( agent, env_kwargs, render_mode, gamma = 1.0, num_rollouts=50, num_workers = 4):
    # progress_bar = tqdm(total=num_rollouts , desc="Episodes", unit="episodes")
    all_rewards = []
    work_queue = list(range(num_rollouts))  # Create a work queue with rollout IDs
    completed_rollouts = 0  # Counter for completed rollouts
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_rollout = {executor.submit(
                                                    worker_rollout, 
                                                    worker_id, 
                                                    agent, 
                                                    render_mode=render_mode,
                                                    env_kwargs=env_kwargs, 
                                                    num_workers=num_workers, 
                                                    num_rollouts=num_rollouts
                                                ):
                                worker_id for worker_id in range(num_workers)}

            while completed_rollouts < num_rollouts:
                # Wait for the next completed rollout and get its result
                for future in concurrent.futures.as_completed(future_to_rollout):
                    worker_id = future_to_rollout[future]
                    try:
                        rewards = future.result()
                        all_rewards.extend(rewards)
                        completed_rollouts += 1

                        if completed_rollouts > num_rollouts:
                            break

                        if work_queue:
                            next_rollout = work_queue.pop(0)
                            future_to_rollout[executor.submit(worker_rollout, worker_id, agent, render_mode=render_mode,
                                                            env_kwargs=env_kwargs, num_workers=num_workers,
                                                            num_rollouts=num_rollouts)] = worker_id
                    except concurrent.futures.process.BrokenProcessPool as e:
                        print("BrokenProcessPool Exception:", e)

    mean_rewards = statistics.mean(all_rewards)
    return mean_rewards



if __name__ == "__main__":
    
    policy_kwargs = dict(
            features_extractor_class=CustomExtractor,
            features_extractor_kwargs=attention_network_kwargs,
        )

    if False:
        optimal_gail_agent, final_gail_agent = retrieve_gail_agents(
                                                                    env= make_configure_env(**env_kwargs).unwrapped, # need only certain parameters
                                                                    artifact_version='trained_model_directory:v2'
                                                                    )
        reward_oracle = final_gail_agent.d
        env_kwargs.update({'reward_oracle':reward_oracle})

    WARM_START = False
    # Get the current date and time
    now = datetime.now()
    month = now.strftime("%m")
    day = now.strftime("%d")
    expert_data_file='expert_data_relative.h5'
    n_cpu =  multiprocessing.cpu_count()
    device = torch.device("cpu")


    def timenow():
        return now.strftime("%H%M")

    
    if   train == TrainEnum.EXPERT_DATA_COLLECTION: # EXPERT_DATA_COLLECTION
        # env = make_configure_env(**env_kwargs).unwrapped
        with open("config.json") as f:
            config = json.load(f)
        device = torch.device("cpu")
        model = PPO(
                    "MlpPolicy", 
                    gym.make(**env_kwargs),
                    policy_kwargs=policy_kwargs,
                    device=device
                    )
        # expert = PPO.load("highway_attention_ppo/model", env=gym.make(**env_kwargs), device=device) # This is not really ultimately treated as expert. Just some policy to run ego.
        wandb.init(project="RL", name="inference")
        # Access the run containing the logged artifact

        # Download the artifact
        artifact = wandb.use_artifact("trained_model:latest")
        artifact_dir = artifact.download()

        # Load the model from the downloaded artifact
        rl_agent_path = os.path.join(artifact_dir, "RL_agent.pth")
        model.policy.load_state_dict(torch.load(rl_agent_path, map_location=device))
        wandb.finish()        
        
        append_key_to_dict_of_dict(env_kwargs,'config','mode','expert')
        collect_expert_data  (
                                    model,
                                    config["num_expert_steps"],
                                    filename=expert_data_file,
                                    **env_kwargs
                                )
        print("collect data complete")
    elif train == TrainEnum.RLTRAIN: # training 
        append_key_to_dict_of_dict(env_kwargs,'config','duration',20)
        env = make_vec_env(
                            make_configure_env, 
                            n_envs=n_cpu, 
                            vec_env_cls=SubprocVecEnv, 
                            env_kwargs=env_kwargs
                          )

        total_timesteps=200*1000
        # Set the checkpoint frequency
        checkpoint_freq = total_timesteps/1000  # Save the model every 10,000 timesteps
        model = PPO(
                    "MlpPolicy", 
                    env,
                    n_steps=512 // n_cpu,
                    batch_size=64,
                    learning_rate=2e-3,
                    policy_kwargs=policy_kwargs,
                    verbose=0,
                    )
        

        checkptcallback = CustomCheckpointCallback(checkpoint_freq, 'checkpoint')  # Create an instance of the custom callback
        # Train the model
        with wandb.init(
                        project="RL", 
                        magic=True,
                        ) as run:
            run.name = f"sweep_{month}{day}_{timenow()}"
            # Create the custom callback
            metrics_callback = CustomMetricsCallback()
            curriculamcallback = CustomCurriculamCallback()

            if WARM_START:
                # Download the artifact in case you want to initialize with pre - trained 
                artifact = wandb.use_artifact("trained_model:v6")
                artifact_dir = artifact.download()

                # Load the model from the downloaded artifact
                rl_agent_path = os.path.join(artifact_dir, "RL_agent.pth")
                # if rl_agent_path in locals():
                model.policy.load_state_dict(torch.load(rl_agent_path))

            training_info = model.learn(
                                        total_timesteps=total_timesteps,
                                        callback=[
                                                    checkptcallback, 
                                                    curriculamcallback,
                                                 ]
                                        )
            
            # Log the model as an artifact in wandb
            torch.save(model.policy.state_dict(), 'RL_agent.pth')
            
            artifact = wandb.Artifact("trained_model", type="model")
            artifact.add_file("RL_agent.pth")
            run.log_artifact(artifact)

        wandb.finish()
        model.save("highway_attention_ppo/model_new")

        # Save the final model
        # model.save("highway_attention_ppo/model")
    elif train == TrainEnum.IRLTRAIN:
        env_kwargs.update({'reward_oracle':None})

        project_name = f"random_env_gail_1"
        device = torch.device("cpu")
        
        # IDM + MOBIL is treated as expert.
        with open("config.json") as f:
            train_config = json.load(f)

        exp_obs, exp_acts, _ = extract_expert_data(expert_data_file)

        
        exp_obs = FloatTensor(exp_obs)
        exp_acts = FloatTensor(exp_acts)

        def train_gail_agent(
                                exp_obs=exp_obs, 
                                exp_acts=exp_acts, 
                                gail_agent_path = None, 
                                env_kwargs = None,
                                train_config = None
                            ):
            env= make_configure_env(**env_kwargs).unwrapped
            state_dim = env.observation_space.high.shape[0]*env.observation_space.high.shape[1]
            action_dim = env.action_space.n
            gail_agent = GAIL(
                                state_dim, 
                                action_dim , 
                                discrete=True, 
                                device=torch.device("cpu"), 
                                **train_config, 
                                # **policy_kwargs, 
                                observation_space= env.observation_space
                             ).to(device=device)
            if gail_agent_path is not None:
                gail_agent.load_state_dict(torch.load(gail_agent_path))
            return gail_agent.train(exp_obs=exp_obs, exp_acts=exp_acts, **env_kwargs)
        
        gail_agent_path = None
        if WARM_START:
            # Download the artifact in case you want to initialize with pre - trained 
            artifact = wandb.use_artifact("trained_model:v7")
            artifact_dir = artifact.download()
            # Load the model from the downloaded artifact
            gail_agent_path = os.path.join(artifact_dir, "optimal_gail_agent.pth")

        run_name = f"sweep_{month}{day}_{timenow()}"
        sweep_id = wandb.sweep(sweep_config, project=project_name)

        def train_sweep(exp_obs, exp_acts, config=None):
            with wandb.init(
                            project=project_name,
                            config=config,
                           ) as run:
                config = run.config
                name = ""
                for key, value in config.items():
                    # Discard first 3 letters from the key
                    key = key[3:]

                    # Format the float value
                    value_str = "{:.2f}".format(value).rstrip('0').rstrip('.') if value and '.' in str(value) else str(value)

                    # Append the formatted key-value pair to the name
                    name += f"{key}_{value_str}_"
                run.name = run_name
                append_key_to_dict_of_dict(env_kwargs,'config','duration',config.duration)
                train_config.update({'gae_gamma':config.gae_gamma})
                train_config.update({'discrm_lr':config.discrm_lr})
                print(" config.duration ", env_kwargs['config']['duration'], " config.gae_gamma ", train_config['gae_gamma'])
                    

                rewards, disc_losses, advs, episode_count =       train_gail_agent(
                                                                                    exp_obs=exp_obs, 
                                                                                    exp_acts=exp_acts, 
                                                                                    gail_agent_path=gail_agent_path, 
                                                                                    env_kwargs = env_kwargs,
                                                                                    train_config = train_config
                                                                                  )

                disc_losses = [ float(l.item()) for l in disc_losses]
                advs = [ float(a.item()) for a in advs]
                rewards = [float(r) for r in rewards]
                # Log rewards against epochs as a single plot
                xs=list(range(len(rewards)))
                run.log({"rewards": wandb.plot.line_series(xs=xs, ys=[rewards] ,title="rewards_vs_epochs",keys = [name])})
                run.log({"disc losses": wandb.plot.line_series(xs=xs, ys=[disc_losses], title="disc losses",keys = [name])})
                run.log({"Mean advantages": wandb.plot.line_series(xs=xs, ys=[advs], title="Mean advantages_vs_epochs",keys = [name])})
                run.log({"Episode_count": wandb.plot.line_series(xs=xs, ys=[episode_count], title="Episode_count_vs_epochs",keys = [name])})

                
                # Create a directory for the models
                os.makedirs("models_archive", exist_ok=True)

                shutil.move("optimal_gail_agent.pth", "models_archive/optimal_gail_agent.pth")
                shutil.move("final_gail_agent.pth", "models_archive/final_gail_agent.pth")

                # Log the model as an artifact in wandb
                artifact = wandb.Artifact("trained_model_directory", type="model_directory")
                artifact.add_dir("models_archive")
                run.log_artifact(artifact)

                    
        # rewards, optimal_agent = train_gail_agent(exp_obs=exp_obs, exp_acts=exp_acts, **env_kwargs)

        wandb.agent(
                     sweep_id=sweep_id, 
                     function=lambda: train_sweep(exp_obs, exp_acts)
                   )
        wandb.finish()
    elif train == TrainEnum.IRLDEPLOY:
        append_key_to_dict_of_dict(env_kwargs,'config','duration',40)
        env_kwargs.update({'reward_oracle':None})
        # env_kwargs.update({'render_mode': None})
        env = make_configure_env(**env_kwargs)
        optimal_gail_agent, final_gail_agent = retrieve_gail_agents(
                                                                    env=env,
                                                                    artifact_version='trained_model_directory:v11',
                                                                    project="random_env_gail_1"
                                                                    )
        num_rollouts = 10
        reward = simulate_with_model(
                                            agent=final_gail_agent, 
                                            env_kwargs=env_kwargs, 
                                            render_mode='human', 
                                            num_workers= min(num_rollouts,n_cpu), 
                                            num_rollouts=num_rollouts
                                    )
        print(" Mean reward ", reward)
    elif train == TrainEnum.RLDEPLOY:
        env = make_configure_env(**env_kwargs,duration=40)
        env_kwargs.update({'reward_oracle':None})
        model = PPO(
                    "MlpPolicy", 
                    env.unwrapped,
                    policy_kwargs=policy_kwargs,
                    device=device
                    )
        with open('highway_attention_ppo/network_hierarchy.txt', 'w') as file:
            file.write("-------------------------- Policy network  ---------------------------------\n")
            write_module_hierarchy_to_file(model.policy, file)
            file.write("-------------------------- Value function ----------------------------------\n")
            write_module_hierarchy_to_file(model.policy.value_net, file)
        
        wandb.init(project="RL", name="inference")
        # Access the run containing the logged artifact

        # Download the artifact
        artifact = wandb.use_artifact("trained_model:latest")
        artifact_dir = artifact.download()

        # Load the model from the downloaded artifact
        rl_agent_path = os.path.join(artifact_dir, "RL_agent.pth")
        model.policy.load_state_dict(torch.load(rl_agent_path, map_location=device))
        wandb.finish()
        
        env.render()
        env.viewer.set_agent_display(functools.partial(display_vehicles_attention, env=env, fe=model.policy.features_extractor))
        gamma = 1.0
        num_rollouts = 10
        for _ in range(num_rollouts):
            obs, info = env.reset()
            done = truncated = False
            cumulative_reward = 0
            while not (done or truncated):
                action, _ = model.predict(obs)
                obs, reward, done, truncated, info = env.step(action)
                cumulative_reward += gamma * reward
                print("speed: ",env.vehicle.speed," ,reward: ", reward, " ,cumulative_reward: ",cumulative_reward)
                env.render()
            print("--------------------------------------------------------------------------------------")
    elif train == TrainEnum.BC:
        env = make_configure_env(**env_kwargs)
        state_dim = env.observation_space.high.shape[0]*env.observation_space.high.shape[1]
        action_dim = env.action_space.n
        exp_obs, exp_acts, exp_dones = extract_expert_data(expert_data_file)
        raw_len = len(exp_acts)
        exp_obs, exp_acts, exp_dones = downsample_most_dominant_class(exp_obs, exp_acts, exp_dones)
        filtered_len = len(exp_acts)
        print(' Expert data length before and after filter ', raw_len, filtered_len)

        # Split the data into training and validation sets
        train_obs, val_obs , train_acts, val_acts, train_dones, val_dones= \
                            train_test_split(
                                                exp_obs, 
                                                exp_acts,
                                                exp_dones,
                                                test_size=0.2, 
                                                random_state=42
                                            )

        def transitions(train_obs, train_acts, train_dones):
            transitions = types.Transitions(
                                                obs=np.array(train_obs[:-1]),
                                                acts=np.array(train_acts[:-1]),
                                                dones=np.array(train_dones[:-1]),
                                                infos=np.array(len(train_dones[:-1])*[{}]),
                                                next_obs=np.array(train_obs[1:])
                                            )
            return transitions
        
        num_samples = len(train_acts)
        permuted_indices = np.random.permutation(num_samples)
    # Shuffle all lists using the same index permutation
        shuffled_obs = [train_obs[i] for i in permuted_indices]
        shuffled_acts = [train_acts[i] for i in permuted_indices]
        shuffled_dones = [train_dones[i] for i in permuted_indices]
        training_transitions = transitions(shuffled_obs, shuffled_acts, shuffled_dones)
        rng=np.random.default_rng()
        device = torch.device("cuda")
        policy = DefaultActorCriticPolicy(env, device)
        bc_trainer = bc.BC(
                            observation_space=env.observation_space,
                            action_space=env.action_space,
                            demonstrations=training_transitions,
                            rng=rng,
                            batch_size=32,
                            device = device,
                            policy=policy
                          )
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        reward_before_training, std_reward_before_training = evaluate_policy(bc_trainer.policy, env, 10)
        print(f"Reward before training: {reward_before_training}, std_reward_before_training: {std_reward_before_training}")

        bc_trainer.train(n_epochs=100)
        reward_after_training, std_reward_after_training = evaluate_policy(bc_trainer.policy, env, 10)
        print(f"Reward after training: {reward_after_training}, std_reward_after_training: {std_reward_after_training}") 


        with wandb.init(
                            project="BC", 
                            magic=True,
                        ) as run:
                        run.name = f"sweep_{month}{day}_{timenow()}"
                        # Log the model as an artifact in wandb
                        torch.save(bc_trainer.policy.state_dict(), 'BC_agent.pth')            
                        artifact = wandb.Artifact("trained_model", type="model")
                        artifact.add_file("BC_agent.pth")
                        run.log_artifact(artifact)
        wandb.finish()

        # Iterate through the validation data and make predictions
        with torch.no_grad():
            predicted_labels = [bc_trainer.policy.predict(obs)[0] for obs in val_obs]
            true_labels = val_acts

        # Calculate evaluation metrics
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels, average='weighted')
        recall = recall_score(true_labels, predicted_labels, average='weighted')
        f1 = f1_score(true_labels, predicted_labels, average='weighted')
        conf_matrix = confusion_matrix(true_labels, predicted_labels)

        # Print the metrics
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)


        with torch.no_grad():
            predicted_labels = [bc_trainer.policy.predict(obs)[0] for obs in train_obs]
            true_labels = train_acts

        # Calculate evaluation metrics
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels, average='weighted')
        recall = recall_score(true_labels, predicted_labels, average='weighted')
        f1 = f1_score(true_labels, predicted_labels, average='weighted')

        # Print the Training metrics
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)


        from highway_env.envs.common.action import DiscreteMetaAction
        ACTIONS_ALL = DiscreteMetaAction.ACTIONS_ALL
        plt.figure(figsize=(8, 6))
        class_labels = [ ACTIONS_ALL[idx] for idx in range(len(ACTIONS_ALL))]
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title("Confusion Matrix")
        plt.show()
    elif train == TrainEnum.BCDEPLOY:
        env_kwargs.update({'reward_oracle':None})
        env_kwargs.update({'render_mode': 'human'})
        append_key_to_dict_of_dict(env_kwargs,'config','real_time_rendering',True)
        env = make_configure_env(**env_kwargs)
        rng=np.random.default_rng()
        device = torch.device("cpu")
        policy = DefaultActorCriticPolicy(env, device)
        bc_trainer = bc.BC(
                            observation_space=env.observation_space,
                            action_space=env.action_space,
                            rng=rng,
                            device = device,
                            policy=policy
                          )
        policy = bc_trainer.policy
        wandb.init(project="BC", name="inference")
        # Access the run containing the logged artifact

        # Download the artifact
        artifact = wandb.use_artifact("trained_model:latest")
        artifact_dir = artifact.download()

        # Load the model from the downloaded artifact
        rl_agent_path = os.path.join(artifact_dir, "BC_agent.pth")
        policy.load_state_dict(torch.load(rl_agent_path, map_location=device))
        wandb.finish()
        num_rollouts = 10
        gamma = 1.0
        # reward = simulate_with_model(
        #                                 agent=policy, 
        #                                 env_kwargs=env_kwargs, 
        #                                 render_mode='human', 
        #                                 num_workers= min(num_rollouts, 1), 
        #                                 num_rollouts=num_rollouts
        #                             )
        # print(" Mean reward ", reward)
        for _ in range(num_rollouts):
            obs, info = env.reset()
            done = truncated = False
            cumulative_reward = 0
            while not (done or truncated):
                action, _ = policy.predict(obs)
                obs, reward, done, truncated, info = env.step(action)
                cumulative_reward += gamma * reward
                print("speed: ",env.vehicle.speed," ,reward: ", reward, " ,cumulative_reward: ",cumulative_reward)
                env.render()
            print("--------------------------------------------------------------------------------------")

