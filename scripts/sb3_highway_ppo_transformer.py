import torch.multiprocessing as mp
import functools
import gymnasium as gym
import seaborn as sns
from stable_baselines3 import PPO
import torch
import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from sb3_contrib import  RecurrentPPO
import os
import json
import wandb
from datetime import datetime
import time
from torch import FloatTensor
import shutil
from models.gail import GAIL
from generate_expert_data import expert_data_collector, retrieve_agent
from forward_simulation import make_configure_env, append_key_to_dict_of_dict, simulate_with_model
from sb3_callbacks import CustomCheckpointCallback, CustomMetricsCallback, CustomCurriculamCallback
from utilities import *
from utils import record_videos
import warnings
from imitation.algorithms import bc
from python_config import sweep_config, env_kwargs, TrainEnum, train
import matplotlib.pyplot as plt
from imitation.algorithms import bc
import importlib
import pandas as pd
import tracemalloc
from highway_env.envs.common.observation import *
warnings.filterwarnings("ignore")

from highway_env.envs.common.action import DiscreteMetaAction
ACTIONS_ALL = DiscreteMetaAction.ACTIONS_ALL





attention_network_kwargs = dict(
    # in_size=5*15,
    embedding_layer_kwargs={"in_size": 7, "layer_sizes": [64, 64], "reshape": False},
    attention_layer_kwargs={
                                "feature_size": 64, 
                                "heads": 2, 
                                # "dropout_factor" :0.2
                           },
    # num_layers = 3,
)

def timenow():
    return now.strftime("%H%M")


def print_stack_size():
    while True:
        current_size, peak_size = tracemalloc.get_traced_memory()
        print(f"Current stack size: {current_size / (1024 * 1024):.2f} MB")
        time.sleep(15)  # Adjust the interval as needed
# ==================================
#        Main script  20 
# ==================================


import threading
if __name__ == "__main__":
    tracemalloc.start()  # Start memory tracing
    # Create a multiprocessing process to periodically print stack size
    stack_size_thread = threading.Thread(target=print_stack_size)
    stack_size_thread.daemon = True
    stack_size_thread.start()

    DAGGER = True
    # policy_kwargs = dict(
    #         # policy=MLPPolicy,
    #         features_extractor_class=CustomExtractor,
    #         features_extractor_kwargs=attention_network_kwargs,
    #     )
    policy_kwargs = dict(
                            features_extractor_class=CustomImageExtractor,
                            features_extractor_kwargs=dict(hidden_dim=64),
                        )


    WARM_START = False
    # Get the current date and time
    now = datetime.now()
    month = now.strftime("%m")
    day = now.strftime("%d")
    zip_filename = 'expert_data_trial.zip'
    n_cpu =  mp.cpu_count()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    extract_path = 'data'

    import python_config
    importlib.reload(python_config)
    from python_config import sweep_config
    print(sweep_config['parameters'])

    batch_size = sweep_config['parameters']['batch_size']['values'][0]
    num_epochs = sweep_config['parameters']['num_epochs']['values'][0]
    num_deploy_rollouts = 5

    try:
        if   train == TrainEnum.EXPERT_DATA_COLLECTION: # EXPERT_DATA_COLLECTION
            project= f"BC_1"                           
            oracle_agent                            = retrieve_agent(
                                                            artifact_version='trained_model_directory:latest',
                                                            agent_model = 'agent_final.pt',
                                                            device=device,
                                                            project=project
                                                        )
            append_key_to_dict_of_dict(env_kwargs,'config','mode','MDPVehicle')
            append_key_to_dict_of_dict(env_kwargs,'config','deploy',True)
            policy = DefaultActorCriticPolicy(make_configure_env(**env_kwargs).unwrapped, device, **policy_kwargs)
            policy.load_state_dict(oracle_agent.state_dict())
            policy.eval()
            expert_data_collector(  
                                    policy,
                                    extract_path = extract_path,
                                    zip_filename=zip_filename,
                                    delta_iterations = 2,
                                    **{**env_kwargs, **{'expert':'MDPVehicle'}}           
                                )
            print(" finished collecting data for ALL THE files ")
        elif train == TrainEnum.RLTRAIN: # training  # Reinforcement learning with curriculam update 
            env_kwargs.update({'reward_oracle':None})
            append_key_to_dict_of_dict(env_kwargs,'config','duration',10)
            append_key_to_dict_of_dict(env_kwargs,'config','EGO_LENGTH',8)
            append_key_to_dict_of_dict(env_kwargs,'config','EGO_WIDTH',4)
            append_key_to_dict_of_dict(env_kwargs,'config','max_vehicles_count',40)
            env = make_vec_env(
                                make_configure_env, 
                                n_envs=n_cpu, 
                                vec_env_cls=SubprocVecEnv, 
                                env_kwargs=env_kwargs
                            )

            total_timesteps=100*1000
            # Set the checkpoint frequency
            checkpoint_freq = total_timesteps/1000  # Save the model every 10,000 timesteps

            # policy = CustomMLPPolicy(env.observation_space, env.action_space)
            model = PPO(
                            'MlpPolicy',
                            env,
                            n_steps=2048 // n_cpu,
                            batch_size=32,
                            learning_rate=2e-3,
                            policy_kwargs=policy_kwargs,
                            # device="cpu",
                            verbose=1,
                        )
            

            checkptcallback = CustomCheckpointCallback(checkpoint_freq, 'checkpoint')  # Create an instance of the custom callback
            project= f"RL"
            run_name = f"sweep_{month}{day}_{timenow()}"
            # Create the custom callback
            metrics_callback = CustomMetricsCallback()
            curriculamcallback = CustomCurriculamCallback()


            training_info = model.learn(
                                        total_timesteps=total_timesteps,
                                        callback=[
                                                    checkptcallback, 
                                                    curriculamcallback,
                                                ]
                                        )
                

            save_checkpoint(
                                project = project, 
                                run_name=run_name,
                                epoch = None, 
                                model = model.policy,
                            )
            # model.save("highway_attention_ppo/model_new")

            # Save the final model
            # model.save("highway_attention_ppo/model")
        elif train == TrainEnum.IRLTRAIN:
            env_kwargs.update({'reward_oracle':None})
            project_name = f"random_env_gail_1"
            
            
            # IDM + MOBIL is treated as expert.
            with open("config.json") as f:
                train_config = json.load(f)


            train_data_loader                                              = create_dataloaders(
                                                                                                    zip_filename,
                                                                                                    extract_path, 
                                                                                                    device=device,
                                                                                                    batch_size=batch_size,
                                                                                                    n_cpu = n_cpu
                                                                                                )
            train_data_loaders = [train_data_loader]
            def train_gail_agent(
                                    gail_agent_path = None, 
                                    env_kwargs = None,
                                    train_config = None,
                                    train_data_loaders=train_data_loaders,
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

                gail_agent.train( 
                                            data_loaders=train_data_loaders,
                                            **env_kwargs
                                )
                return gail_agent
            
            gail_agent_path = None


            run_name = f"sweep_{month}{day}_{timenow()}"
            sweep_id = wandb.sweep(sweep_config, project=project_name)



            trained_gail_agent                        =       train_gail_agent(
                                                                                    gail_agent_path=None, 
                                                                                    env_kwargs = env_kwargs,
                                                                                    train_config = train_config,
                                                                                    train_data_loaders=train_data_loaders
                                                                                )
            expert_data_collector(
                                    trained_gail_agent.pi, # This is the exploration policy
                                    extract_path = extract_path,
                                    zip_filename=zip_filename,
                                    delta_iterations = 1,
                                    **{
                                        **env_kwargs, 
                                        **{'expert':'MDPVehicle'}
                                        }           
                                )
        elif train == TrainEnum.IRLDEPLOY:
            append_key_to_dict_of_dict(env_kwargs,'config','duration',40)
            append_key_to_dict_of_dict(env_kwargs,'config','vehicles_count',150)
            append_key_to_dict_of_dict(env_kwargs,'config','deploy',True)
            append_key_to_dict_of_dict(env_kwargs,'config','real_time_rendering',True)
            env_kwargs.update({'reward_oracle':None})
            append_key_to_dict_of_dict(env_kwargs,'config','mode',None)
            # env_kwargs.update({'render_mode': None})
            env = make_configure_env(**env_kwargs)
            record_videos(env=env, name_prefix = 'GAIL', video_folder='videos/GAIL')
            artifact_version= f'trained_model_directory:latest',
            agent_model = f'final_gail_agent.pth',
            project= f"random_env_gail_1"
            optimal_gail_agent                       = retrieve_agent(
                                                                        artifact_version = artifact_version,
                                                                        agent_model = agent_model,
                                                                        device=device,
                                                                        project = project
                                                                    )
            num_rollouts = 10
            # reward = simulate_with_model(
            #                                     agent=optimal_gail_agent, 
            #                                     env_kwargs=env_kwargs, 
            #                                     render_mode=None, 
            #                                     num_workers= 1, 
            #                                     num_rollouts=num_deploy_rollouts
            #                             )
            # print(" Mean reward ", reward)
            gamma = 1.0
            env.render()
            agent = optimal_gail_agent
            # env.viewer.set_agent_display(functools.partial(display_vehicles_attention, env=env, fe=agent.policy.features_extractor))
            for _ in range(num_deploy_rollouts):
                obs, info = env.reset()
                env.step(4)
                done = truncated = False
                cumulative_reward = 0
                while not (done or truncated):
                    action = agent.act(obs.flatten())
                    env.vehicle.actions = []
                    obs, reward, done, truncated, info = env.step(action)
                    cumulative_reward += gamma * reward
                    env.render()
                print("speed: ",env.vehicle.speed," ,reward: ", reward, " ,cumulative_reward: ",cumulative_reward)
                print("--------------------------------------------------------------------------------------")
        elif train == TrainEnum.RLDEPLOY:
            append_key_to_dict_of_dict(env_kwargs,'config','vehicles_count',30)
            append_key_to_dict_of_dict(env_kwargs,'config','deploy',True)
            append_key_to_dict_of_dict(env_kwargs,'config','real_time_rendering',True)
            append_key_to_dict_of_dict(env_kwargs,'config','duration',80)
            append_key_to_dict_of_dict(env_kwargs,'config','offscreen_rendering',False)
            env_kwargs.update({'reward_oracle':None})
            env = make_configure_env(**env_kwargs)
            env = record_videos(env=env, name_prefix = 'RL', video_folder='videos/RL')
            # RL_agent                            = retrieve_agent(
            #                                                     artifact_version='trained_model_directory:latest',
            #                                                     agent_model = 'RL_agent_final.pth',
                                                                #   device = device,
            #                                                     project="RL"
            #                                                     )
            
            wandb.init(project="RL", name="inference")
            # Access the run containing the logged artifact

            # Download the artifact
            artifact = wandb.use_artifact("trained_model:latest")
            artifact_dir = artifact.download()

            # Load the model from the downloaded artifact
            rl_agent_path = os.path.join(artifact_dir, "RL_agent.pth")
            model = torch.load(rl_agent_path, map_location=device)
            wandb.finish()
            
            print(model)
            env.render()
            env.viewer.set_agent_display(
                                            functools.partial(
                                                                display_vehicles_attention, 
                                                                env=env, 
                                                                fe=model.features_extractor,
                                                                device=device
                                                            )
                                        )
            gamma = 1.0
            for _ in range(num_deploy_rollouts):
                obs, info = env.reset()
                env.step(4)
                done = truncated = False
                cumulative_reward = 0
            
                while not (done or truncated):
                    start_time = time.time()
                    action, _ = model.predict(obs)
                    env.vehicle.actions = []
                    obs, reward, done, truncated, info = env.step(action)
                    cumulative_reward += gamma * reward
                    env.render()
                    end_time = time.time()
                print("speed: ",env.vehicle.speed," ,reward: ", reward, " ,cumulative_reward: ",cumulative_reward)
                print("--------------------------------------------------------------------------------------")
        elif train == TrainEnum.BC:
            env_kwargs.update({'reward_oracle':None})
            append_key_to_dict_of_dict(env_kwargs,'config','deploy',True)
            env = make_configure_env(**env_kwargs)
            env=env.unwrapped
            state_dim = env.observation_space.high.shape[0]*env.observation_space.high.shape[1]
            action_dim = env.action_space.n
            rng=np.random.default_rng()
            project = "BC_1"
            policy = DefaultActorCriticPolicy(env, device, **policy_kwargs)
            print("Default policy initialized ")
            run_name = f"sweep_{month}{day}_{timenow()}"
            # sweep_id = wandb.sweep(sweep_config, project=project_name)
            
            metrics_plot_path = "" #f"{extract_path}/metrics.png"

            def create_trainer(env, policy, device=device, **kwargs):
                return       bc.BC(
                                    observation_space=env.observation_space,
                                    action_space=env.action_space,
                                    demonstrations=None, #training_transitions,
                                    rng=np.random.default_rng(),
                                    batch_size=kwargs['batch_size'],
                                    device = device,
                                    policy=policy
                                    )        
        

            def _train(env, policy, zip_filename, extract_path, device=device, **training_kwargs):
                num_epochs = training_kwargs['num_epochs']
                checkpoint_interval = num_epochs//2
                append_key_to_dict_of_dict(env_kwargs,'config','mode','MDPVehicle')
                _validation_metrics =       {
                                                "accuracy"  : [], 
                                                "precision" : [], 
                                                "recall"    : [],
                                                "f1"        : []
                                            }
                trainer = create_trainer(env, policy, batch_size=batch_size, num_epochs=num_epochs, device=device) # Unfotunately needed to instantiate repetitively
                print(" trainer policy (train_mode ?)", trainer.policy.training)
                epoch = None
                train_datasets = []
                visited_data_files = set([])
                for epoch in range(num_epochs): # Epochs here correspond to new data distribution (as maybe collecgted through DAGGER)
                    print(f'Loadng training data loader for epoch {epoch}')
                    # train_data_loader                                            = create_dataloaders(
                    #                                                                                       zip_filename,
                    #                                                                                       train_datasets, 
                    #                                                                                       device=device,
                    #                                                                                       batch_size=training_kwargs['batch_size'],
                    #                                                                                       n_cpu = n_cpu,
                    #                                                                                       visited_data_files=visited_data_files
                    #                                                                                   )
                    custom_data_loader = CustomDataLoader(zip_filename, device, visited_data_files, batch_size, n_cpu)
                    print(f'Loaded training data loader for epoch {epoch}')
                    last_epoch = (epoch ==num_epochs-1)
                    num_mini_batches = 500000 if last_epoch else 2500 # Mini epoch here correspond to typical epoch
                    trainer.set_demonstrations(custom_data_loader)
                    print(f'Beginning Training for epoch {epoch}')
                    # with torch.autograd.detect_anomaly():
                    trainer.train(n_batches=num_mini_batches)
                    # del train_data_loader
                    print(f'Ended training for epoch {epoch}')

                    policy = trainer.policy
                    policy.eval()  
                    if not last_epoch and DAGGER:
                        print(f'Began Dagger data collection for epoch {epoch}')
                        expert_data_collector(
                                                policy, # This is the exploration policy
                                                extract_path = extract_path,
                                                zip_filename=zip_filename,
                                                delta_iterations = 1,
                                                **{
                                                    **env_kwargs, 
                                                    **{'expert':'MDPVehicle'}
                                                    }           
                                            )
                        print(f'End Dagger data collection for epoch {epoch}')

                    if checkpoint_interval !=0 and epoch % checkpoint_interval == 0 and not last_epoch:
                        print(f"saving check point for epoch {epoch}", epoch)
                        torch.save(policy , f"models_archive/BC_agent_{epoch}.pth")
                    if metrics_plot_path:
                        print(f'Beginning validation for epoch {epoch}')
                        accuracy, precision, recall, f1 = calculate_validation_metrics(
                                                                                        policy, 
                                                                                        zip_filename=zip_filename,
                                                                                        plot_path=f"heatmap_{epoch}.png" 
                                                                                    )
                        _validation_metrics["accuracy"].append(accuracy)
                        _validation_metrics["precision"].append(precision)
                        _validation_metrics["recall"].append(recall)
                        _validation_metrics["f1"].append(f1)
                        print(f'End validation for epoch {epoch}')
                    policy.to(device)
                    policy.train()
                



                if metrics_plot_path:
                    # Plotting metrics
                    plt.figure(figsize=(10, 6))
                    epochs = range(num_epochs)
                    for metric_name, metric_values in _validation_metrics.items():
                        plt.plot(epochs, metric_values, label=metric_name)
                        plt.xlabel("Epochs")
                        plt.ylabel("Metrics Value")
                        plt.title("Validation Metrics over Epochs")
                        plt.legend()
                        plt.grid(True)
                        plt.savefig(f"{extract_path}/metrics.png")
                    
                return trainer  

            

            
            bc_trainer                                  = _train(
                                                                    env,
                                                                    policy,
                                                                    zip_filename,
                                                                    extract_path,
                                                                    device=device,
                                                                    num_epochs=num_epochs, 
                                                                    batch_size=batch_size,
                                                                )
            final_policy = bc_trainer.policy
            final_policy.to(torch.device('cpu'))
            final_policy.eval()
            print('Saving final model')
            save_checkpoint(
                                project = project, 
                                run_name=run_name,
                                epoch = None, 
                                model = final_policy,
                                metrics_plot_path = metrics_plot_path
                            )
            print('Saved final model')

            print('Beginnig final validation step')
            accuracy, precision, recall, f1 = calculate_validation_metrics(
                                                                            final_policy, 
                                                                            zip_filename=zip_filename, 
                                                                            # plot_path=f"heatmap_{epoch}.png" 
                                                                            )
            print('Ending final validation step and plotting the heatmap ')
        elif train == TrainEnum.BCDEPLOY:
            env_kwargs.update({'reward_oracle':None})
            # env_kwargs.update({'render_mode': 'human'})
            append_key_to_dict_of_dict(env_kwargs,'config','max_vehicles_count',75)
            append_key_to_dict_of_dict(env_kwargs,'config','real_time_rendering',True)
            append_key_to_dict_of_dict(env_kwargs,'config','deploy',True)
            append_key_to_dict_of_dict(env_kwargs,'config','duration',40)
            append_key_to_dict_of_dict(env_kwargs,'config','offscreen_rendering',False)
            env = make_configure_env(**env_kwargs)
            env = record_videos(env=env, name_prefix = 'BC', video_folder='videos/BC')
            BC_agent                            = retrieve_agent(
                                                                    artifact_version='trained_model_directory:latest',
                                                                    agent_model = 'agent_final.pt',
                                                                    device=device,
                                                                    project="BC_1"
                                                                )
            gamma = 1.0
            env.render()
            # try:
            #     env.viewer.set_agent_display(
            #                                     functools.partial(
            #                                                         display_vehicles_attention, 
            #                                                         env=env, 
            #                                                         fe=BC_agent.features_extractor,
            #                                                         device=device
            #                                                     )
            #                                 )
            # except:
            #     pass
            policy = DefaultActorCriticPolicy(env, device, **policy_kwargs)
            policy.load_state_dict(BC_agent.state_dict())
            # policy.eval()
            image_space_obs = isinstance(env.observation_type,GrayscaleObservation)
            if image_space_obs:   
                fig = plt.figure(figsize=(8, 16))
                import cv2
            for _ in range(num_deploy_rollouts):
                obs, info = env.reset()
                env.step(4)
                done = truncated = False
                cumulative_reward = 0
                while not (done or truncated):
                    action, _ = policy.predict(obs)
                    env.vehicle.actions = []
                    obs, reward, done, truncated, info = env.step(action)
                    cumulative_reward += gamma * reward
                    if image_space_obs:
                        for i in range(1):
                            image = obs[3,:]
                            input_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                            # action_logits = policy.action_net(input_tensor)
                            action_logits = policy.action_net(policy.mlp_extractor(policy.features_extractor(torch.Tensor(obs).unsqueeze(0)))[0])
                            selected_action = torch.argmax(action_logits, dim=1)
                            action_logits[0, selected_action].backward()
                            gradients = input_tensor.grad.squeeze()
                            # Normalize and overlay the gradient-based heatmap on the original image
                            heatmap = gradients.numpy()
                            heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))  # Normalize between 0 and 1
                            heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
                            heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
                            result  = cv2.addWeighted(image.astype(np.uint8), 0.5, heatmap, 0.5, 0)
                            plt.imshow(result, cmap='gray', origin='lower', aspect = 0.5)
                            plt.xlim(20, 40)
                            plt.show(block=False)
                            # plt.pause(0.01)
                    plt.pause(0.01)
                    env.render()
                print("speed: ",env.vehicle.speed," ,reward: ", reward, " ,cumulative_reward: ",cumulative_reward)
                print("--------------------------------------------------------------------------------------")
        elif train == TrainEnum.ANALYSIS:
            train_data_loader                                             = create_dataloaders(
                                                                                                    zip_filename,
                                                                                                    extract_path, 
                                                                                                    device=device,
                                                                                                    batch_size=batch_size,
                                                                                                    n_cpu = n_cpu
                                                                                                )
            # Create a DataFrame from the data loader
            data_list = np.empty((0, 70))
            actions = []
            for batch in train_data_loader:
                whole_batch_states = batch['obs'].reshape(-1, 70) 
                actions.extend(batch['acts'].numpy().astype(int))
                data_list = np.vstack((data_list, whole_batch_states.numpy()))
            data_df = pd.DataFrame(data_list)
            actions = np.array(actions)
            action_counts = np.bincount(actions.astype(int))

            # Create a bar chart for action distribution
            plt.figure(figsize=(10, 6))
            plt.bar(range(len(action_counts)), action_counts, tick_label=range(len(action_counts)))
            plt.xlabel("Action")
            plt.ylabel("Frequency")
            plt.title("Action Distribution")
            # plt.show()

            # Define the ranges for each feature
            feature_ranges = np.linspace(-1, 1, num=101)  # Adjust the number of bins as needed

            # Calculate the count of samples within each range for each feature
            sample_counts = np.array([[((data_df[col] >= feature_ranges[i]) & (data_df[col] <= feature_ranges[i+1])).sum()
                                    for i in range(len(feature_ranges)-1)]
                                    for col in data_df.columns])
            sample_counts = sample_counts.T
            # Normalize the sample counts to a range between 0 and 1
            normalized_counts = (sample_counts - sample_counts.min()) / (sample_counts.max() - sample_counts.min())
            # Reshape the sample counts to create a heatmap
            # sample_count_matrix = sample_counts.reshape(-1, 1)

            # Create a color map based on the sample counts
            cmap = sns.cubehelix_palette(start=2, rot=0, dark=0, light=1.0, reverse=False, as_cmap=True)


            # Create a violin plot directly from the data loader
            # Create a figure with two subplots
            fig, axes = plt.subplots(2, 1, figsize=(12, 6))
            sns.violinplot(data=data_df, inner="quartile", ax=axes[0])
            axes[0].set_title("Violin Plot (input)")
            sns.heatmap(data=sample_counts, cmap=cmap, cbar=False, ax=axes[1])
            axes[1].set_title("Heatmap (input)")

            plt.tight_layout()
            plt.show()
    except KeyboardInterrupt:
        tracemalloc.stop()  # Stop memory tracing when done