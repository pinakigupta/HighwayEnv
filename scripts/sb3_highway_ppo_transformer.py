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
from enum import Enum
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
from python_config import sweep_config, env_kwargs
import matplotlib.pyplot as plt
from imitation.algorithms import bc
import importlib
import pandas as pd
import random

warnings.filterwarnings("ignore")

from highway_env.envs.common.action import DiscreteMetaAction
ACTIONS_ALL = DiscreteMetaAction.ACTIONS_ALL


class TrainEnum(Enum):
    RLTRAIN = 0
    RLDEPLOY = 1
    IRLTRAIN = 2
    IRLDEPLOY = 3
    EXPERT_DATA_COLLECTION =4
    BC = 5
    BCDEPLOY = 6
    ANALYSIS = 7

train = TrainEnum.RLDEPLOY



attention_network_kwargs = dict(
    # in_size=5*15,
    embedding_layer_kwargs={"in_size": 7, "layer_sizes": [64, 64], "reshape": False},
    attention_layer_kwargs={"feature_size": 64, "heads": 2},
    # num_layers = 3,
)

def timenow():
    return now.strftime("%H%M")



# ==================================
#        Main script  20 
# ==================================



if __name__ == "__main__":
    
    DAGGER = True
    policy_kwargs = dict(
            # policy=MLPPolicy,
            features_extractor_class=CustomExtractor,
            features_extractor_kwargs=attention_network_kwargs,
        )

    if False:
        optimal_gail_agent                       = retrieve_agent(
                                                                    # env= make_configure_env(**env_kwargs).unwrapped, # need only certain parameters
                                                                    artifact_version='trained_model_directory:v2',
                                                                    agent_model='optimal_gail_agent.pth'
                                                                    )
        final_gail_agent                         = retrieve_agent(
                                                                    # env= make_configure_env(**env_kwargs).unwrapped, # need only certain parameters
                                                                    artifact_version='trained_model_directory:v2',
                                                                    agent_model='final_gail_agent.pth'
                                                                    )
        reward_oracle = final_gail_agent.d
        env_kwargs.update({'reward_oracle':reward_oracle})

    WARM_START = False
    # Get the current date and time
    now = datetime.now()
    month = now.strftime("%m")
    day = now.strftime("%d")
    zip_filename = 'expert_data.zip'
    n_cpu =  mp.cpu_count()
    device = torch.device("cpu")
    extract_path = 'data'

    import python_config
    importlib.reload(python_config)
    from python_config import sweep_config
    print(sweep_config['parameters'])

    batch_size = sweep_config['parameters']['batch_size']['values'][0]
    num_epochs = sweep_config['parameters']['num_epochs']['values'][0]

    
    if   train == TrainEnum.EXPERT_DATA_COLLECTION: # EXPERT_DATA_COLLECTION
        oracle_agent                       = retrieve_agent(
                                                                artifact_version='trained_model_directory:latest',
                                                                agent_model = 'optimal_gail_agent.pth',
                                                                project="random_env_gail_1",
                                                             )
        append_key_to_dict_of_dict(env_kwargs,'config','mode','expert')
        append_key_to_dict_of_dict(env_kwargs,'config','deploy',True)
        expert_data_collector(  
                                oracle_agent,
                                extract_path = extract_path,
                                zip_filename=zip_filename,
                                delta_iterations = 10,
                                **{**env_kwargs, **{'expert':None}}           
                             )
    elif train == TrainEnum.RLTRAIN: # training  # Reinforcement learning with curriculam update 
        env_kwargs.update({'reward_oracle':None})
        append_key_to_dict_of_dict(env_kwargs,'config','duration',10)
        append_key_to_dict_of_dict(env_kwargs,'config','EGO_LENGTH',8)
        append_key_to_dict_of_dict(env_kwargs,'config','EGO_WIDTH',4)
        env = make_vec_env(
                            make_configure_env, 
                            n_envs=n_cpu*3, 
                            vec_env_cls=SubprocVecEnv, 
                            env_kwargs=env_kwargs
                          )

        total_timesteps=200*10000
        # Set the checkpoint frequency
        checkpoint_freq = total_timesteps/1000  # Save the model every 10,000 timesteps
        model = RecurrentPPO(
                        'MlpLstmPolicy',
                        env,
                        n_steps=2048 // n_cpu,
                        batch_size=32,
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
            torch.save(model.policy, 'RL_agent.pth')
            
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

        # expert_data_file = "expert_train_data_0.h5"


        # exp_obs = FloatTensor(exp_obs)
        # exp_acts = FloatTensor(exp_acts)
        train_data_loader, _ , _                                     = create_dataloaders(
                                                                                                zip_filename,
                                                                                                extract_path, 
                                                                                                device=device,
                                                                                                batch_size=batch_size,
                                                                                                n_cpu = n_cpu
                                                                                            )
        train_data_loaders = [train_data_loader]
        def train_gail_agent(
                                # exp_obs=exp_obs, 
                                # exp_acts=exp_acts,
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
            if gail_agent_path is not None:
                gail_agent.load_state_dict(torch.load(gail_agent_path))
            return gail_agent.train(
                                        # exp_obs=exp_obs, 
                                        # exp_acts=exp_acts, 
                                        data_loaders=train_data_loaders,
                                        **env_kwargs
                                    )
        
        gail_agent_path = None
        if WARM_START:
            # Download the artifact in case you want to initialize with pre - trained 
            artifact = wandb.use_artifact("trained_model:v7")
            artifact_dir = artifact.download()
            # Load the model from the downloaded artifact
            gail_agent_path = os.path.join(artifact_dir, "optimal_gail_agent.pth")

        run_name = f"sweep_{month}{day}_{timenow()}"
        sweep_id = wandb.sweep(sweep_config, project=project_name)


        rewards, disc_losses, advs, episode_count =       train_gail_agent(
                                                                                gail_agent_path=None, 
                                                                                env_kwargs = env_kwargs,
                                                                                train_config = train_config,
                                                                                train_data_loaders=train_data_loaders
                                                                            )
        
        # def train_sweep(data_loaders, config=None):
        #     with wandb.init(
        #                     project=project_name,
        #                     config=config,
        #                    ) as run:
        #         config = run.config
        #         name = ""
        #         for key, value in config.items():
        #             # Discard first 3 letters from the key
        #             key = key[3:]

        #             # Format the float value
        #             value_str = "{:.2f}".format(value).rstrip('0').rstrip('.') if value and '.' in str(value) else str(value)

        #             # Append the formatted key-value pair to the name
        #             name += f"{key}_{value_str}_"
        #         run.name = run_name
        #         append_key_to_dict_of_dict(env_kwargs,'config','duration',config.duration)
        #         train_config.update({'gae_gamma':config.gae_gamma})
        #         train_config.update({'discrm_lr':config.discrm_lr})
        #         print(" config.duration ", env_kwargs['config']['duration'], " config.gae_gamma ", train_config['gae_gamma'])
                    

        #         rewards, disc_losses, advs, episode_count =       train_gail_agent(
        #                                                                             # exp_obs=exp_obs, 
        #                                                                             # exp_acts=exp_acts,
        #                                                                             zip_filename ,
        #                                                                             gail_agent_path=gail_agent_path, 
        #                                                                             env_kwargs = env_kwargs,
        #                                                                             train_config = train_config,
        #                                                                             device=device,
        #                                                                             batch_size=batch_size,
        #                                                                             train_data_loaders=data_loaders
        #                                                                           )


        #         disc_losses = [ float(l.item()) for l in disc_losses]
        #         advs = [ float(a.item()) for a in advs]
        #         rewards = [float(r) for r in rewards]
        #         # Log rewards against epochs as a single plot
        #         xs=list(range(len(rewards)))
        #         run.log({"rewards": wandb.plot.line_series(xs=xs, ys=[rewards] ,title="rewards_vs_epochs",keys = [name])})
        #         run.log({"disc losses": wandb.plot.line_series(xs=xs, ys=[disc_losses], title="disc losses",keys = [name])})
        #         run.log({"Mean advantages": wandb.plot.line_series(xs=xs, ys=[advs], title="Mean advantages_vs_epochs",keys = [name])})
        #         run.log({"Episode_count": wandb.plot.line_series(xs=xs, ys=[episode_count], title="Episode_count_vs_epochs",keys = [name])})

                
        #         # Create a directory for the models
        #         clear_and_makedirs("models_archive")

        #         shutil.move("optimal_gail_agent.pth", "models_archive/optimal_gail_agent.pth")
        #         shutil.move("final_gail_agent.pth", "models_archive/final_gail_agent.pth")

        #         # Log the model as an artifact in wandb
        #         artifact = wandb.Artifact("trained_model_directory", type="model_directory")
        #         artifact.add_dir("models_archive")
        #         run.log_artifact(artifact)

                    
        # # rewards, optimal_agent = train_gail_agent(exp_obs=exp_obs, exp_acts=exp_acts, **env_kwargs)

        # wandb.agent(
        #              sweep_id=sweep_id, 
        #              function= lambda: train_sweep(train_data_loaders)
        #            )
        # wandb.finish()
    elif train == TrainEnum.IRLDEPLOY:
        append_key_to_dict_of_dict(env_kwargs,'config','duration',40)
        append_key_to_dict_of_dict(env_kwargs,'config','vehicles_count',150)
        append_key_to_dict_of_dict(env_kwargs,'config','deploy',True)
        append_key_to_dict_of_dict(env_kwargs,'config','real_time_rendering',True)
        env_kwargs.update({'reward_oracle':None})
        append_key_to_dict_of_dict(env_kwargs,'config','mode',None)
        # env_kwargs.update({'render_mode': None})
        env = make_configure_env(**env_kwargs)
        optimal_gail_agent                       = retrieve_agent(
                                                                    # env=env,
                                                                    artifact_version='trained_model_directory:latest',
                                                                    agent_model = 'optimal_gail_agent.pth',
                                                                    project="random_env_gail_1"
                                                                 )
        final_gail_agent                         = retrieve_agent(
                                                                    # env=env,
                                                                    artifact_version='trained_model_directory:latest',
                                                                    agent_model = 'final_gail_agent.pth',
                                                                    project="random_env_gail_1"
                                                                 )
        num_rollouts = 10
        reward = simulate_with_model(
                                            agent=optimal_gail_agent, 
                                            env_kwargs=env_kwargs, 
                                            render_mode='human', 
                                            num_workers= min(num_rollouts,1), 
                                            num_rollouts=num_rollouts
                                    )
        print(" Mean reward ", reward)
    elif train == TrainEnum.RLDEPLOY:
        append_key_to_dict_of_dict(env_kwargs,'config','vehicles_count',30)
        append_key_to_dict_of_dict(env_kwargs,'config','deploy',True)
        append_key_to_dict_of_dict(env_kwargs,'config','real_time_rendering',True)
        append_key_to_dict_of_dict(env_kwargs,'config','duration',80)
        env_kwargs.update({'reward_oracle':None})
        env = make_configure_env(**env_kwargs)
        env = record_videos(env)
        # RL_agent                            = retrieve_agent(
        #                                                     artifact_version='trained_model_directory:latest',
        #                                                     agent_model = 'RL_agent_final.pth',
        #                                                     project="RL"
        #                                                     )
        # with open('highway_attention_ppo/network_hierarchy.txt', 'w') as file:
        #     file.write("-------------------------- Policy network  ---------------------------------\n")
        #     write_module_hierarchy_to_file(model.policy, file)
        #     file.write("-------------------------- Value function ----------------------------------\n")
        #     write_module_hierarchy_to_file(model.policy.value_net, file)
        
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
        env.viewer.set_agent_display(functools.partial(display_vehicles_attention, env=env, fe=model.features_extractor))
        gamma = 1.0
        num_rollouts = 10
        for _ in range(num_rollouts):
            obs, info = env.reset()
            env.step(4)
            done = truncated = False
            cumulative_reward = 0
           
            while not (done or truncated):
                start_time = time.time()
                action, _ = model.predict(obs)
                env.vehicle.actions = []
                # for _ in range(10):
                #     env.vehicle.actions.append(action.astype(int).tolist())
                # action = [key for key, val in ACTIONS_ALL.items() if val == env.vehicle.discrete_action()[0]][0]
                obs, reward, done, truncated, info = env.step(action)
                cumulative_reward += gamma * reward
                env.render()
                end_time = time.time()
                # time.sleep(max(0.0, 1/env.config['simulation_frequency'] - (end_time - start_time))) # smart sleep

            print("speed: ",env.vehicle.speed," ,reward: ", reward, " ,cumulative_reward: ",cumulative_reward)
            print("--------------------------------------------------------------------------------------")
    elif train == TrainEnum.BC:
        env_kwargs.update({'reward_oracle':None})
        append_key_to_dict_of_dict(env_kwargs,'config','deploy',True)
        env = make_configure_env(**env_kwargs)
        state_dim = env.observation_space.high.shape[0]*env.observation_space.high.shape[1]
        action_dim = env.action_space.n
        rng=np.random.default_rng()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        project = "BC_1"
        # device = 'cpu'
        policy = DefaultActorCriticPolicy(env, device, **policy_kwargs)
        print("Default policy initialized ")
        


        project_name = f"BC_1"
        run_name = f"sweep_{month}{day}_{timenow()}"
        # sweep_id = wandb.sweep(sweep_config, project=project_name)

        metrics_plot_path = f"{extract_path}/metrics.png"

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
    

        def _train(zip_filename, extract_path, device=device, **training_kwargs):
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
            print(" trainer policy ", trainer.policy)
            epoch = None
            for epoch in range(num_epochs): # Epochs here correspond to new data distribution (as maybe collecgted through DAGGER)
                train_data_loader                                            = create_dataloaders(
                                                                                                      zip_filename,
                                                                                                      extract_path, 
                                                                                                      device=device,
                                                                                                      batch_size=training_kwargs['batch_size'],
                                                                                                      n_cpu = n_cpu
                                                                                                  )
                
                last_epoch = (epoch ==num_epochs-1)
                num_mini_batches = 150000 if last_epoch else 25000 # Mini epoch here correspond to typical epoch
                trainer.set_demonstrations(train_data_loader)
                trainer.train(n_batches=num_mini_batches)  
                if not last_epoch and DAGGER:
                    expert_data_collector(
                                            trainer.policy, # This is the exploration policy
                                            extract_path = extract_path,
                                            zip_filename=zip_filename,
                                            delta_iterations = 1,
                                            **{
                                                **env_kwargs, 
                                                **{'expert':'MDPVehicle'}
                                                }           
                                        )

                # num_rollouts = 10
                # reward = simulate_with_model(
                #                                     agent=trainer.policy, 
                #                                     env_kwargs=env_kwargs, 
                #                                     render_mode='none', 
                #                                     num_workers= min(num_rollouts,n_cpu), 
                #                                     num_rollouts=num_rollouts
                #                             )
                # print(f"Reward after training epoch {epoch}: {reward}")
                # At the end of each epoch or desired interval
                if checkpoint_interval !=0 and epoch % checkpoint_interval == 0 and not last_epoch:
                    print("saving check point ", epoch)
                    torch.save(trainer , f"models_archive/BC_agent_{epoch}.pth")
                accuracy, precision, recall, f1 = calculate_validation_metrics(
                                                                                trainer, 
                                                                                zip_filename=zip_filename,
                                                                                plot_path=f"heatmap_{epoch}.png" 
                                                                              )
                _validation_metrics["accuracy"].append(accuracy)
                _validation_metrics["precision"].append(precision)
                _validation_metrics["recall"].append(recall)
                _validation_metrics["f1"].append(f1)
            epochs = range(num_epochs)


            accuracy, precision, recall, f1 = calculate_validation_metrics(
                                                                            trainer, 
                                                                            zip_filename=zip_filename, 
                                                                            plot_path=f"heatmap_{epoch}.png" 
                                                                            )


            # Plotting
            plt.figure(figsize=(10, 6))

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
                                                                zip_filename,
                                                                extract_path,
                                                                num_epochs=num_epochs, 
                                                                batch_size=batch_size,
                                                            )
        save_checkpoint(
                            project = project, 
                            run_name=run_name,
                            epoch = None, 
                            trainer = bc_trainer,
                            metrics_plot_path = metrics_plot_path
                        )
        # wandb.finish()        
        # clear_and_makedirs("models_archive")
        # def train_sweep(env, policy, config=None):
        #     with wandb.init(
        #                 project=project_name, 
        #                 config=config,
        #                 mode="disabled"
        #                 # magic=True,
        #             ) as run:
        #         config = run.config
        #         print("config ", config)
        #         run.name = f"sweep_{month}{day}_{timenow()}"
        #         name = ""
        #         for key, value in config.items():
        #             # Discard first 3 letters from the key
        #             key = key[3:]

        #             # Format the float value
        #             value_str = "{:.2f}".format(value).rstrip('0').rstrip('.') if value and '.' in str(value) else str(value)

        #             # Append the formatted key-value pair to the name
        #             name += f"{key}_{value_str}_"
        #         plot_path = f"plot_{name}.png"
        #         batch_size = config.batch_size
        #         num_epochs = config.num_epochs 
        #         bc_trainer = create_trainer(env, policy, batch_size=batch_size, num_epochs=num_epochs)
        #         hdf5_train_file_names, hdf5_val_file_names = _train(
        #                                                              bc_trainer, 
        #                                                              zip_filename,
        #                                                              extract_path,
        #                                                              num_epochs=num_epochs, 
        #                                                              batch_size=batch_size
        #                                                             )
        #         calculate_validation_metrics(bc_trainer, hdf5_train_file_names, hdf5_val_file_names, plot_path=plot_path )

        #         clear_and_makedirs("models_archive")
        #         torch.save(bc_trainer, 'models_archive/BC_agent.pth') 

        #         # Log the model as an artifact in wandb
        #         artifact = wandb.Artifact("trained_model_directory", type="model_directory")
        #         artifact.add_dir("models_archive")
        #         run.log_artifact(artifact)
        #         # Log the saved plot to WandB
        #         run.log({f"plot_{name}": wandb.Image(plot_path)})

        # wandb.agent(
        #             sweep_id=sweep_id, 
        #             function=lambda: train_sweep(env, policy)
        #         )

        # wandb.finish()
    elif train == TrainEnum.BCDEPLOY:
        env_kwargs.update({'reward_oracle':None})
        env_kwargs.update({'render_mode': 'human'})
        append_key_to_dict_of_dict(env_kwargs,'config','max_vehicles_count',150)
        append_key_to_dict_of_dict(env_kwargs,'config','real_time_rendering',True)
        append_key_to_dict_of_dict(env_kwargs,'config','deploy',True)
        append_key_to_dict_of_dict(env_kwargs,'config','duration',100)
        env = make_configure_env(**env_kwargs)
        BC_agent                            = retrieve_agent(
                                                            artifact_version='trained_model_directory:latest',
                                                            agent_model = 'BC_agent_final.pth',
                                                            project="BC_1"
                                                            )
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
        env.viewer.set_agent_display(functools.partial(display_vehicles_attention, env=env, fe=BC_agent.policy.features_extractor))
        for _ in range(num_rollouts):
            obs, info = env.reset()
            done = truncated = False
            cumulative_reward = 0
            with torch.no_grad():
                BC_agent.policy.mlp_extractor.policy_net.eval()
                BC_agent.policy.action_net.eval()
                while not (done or truncated):
                    action, _ = BC_agent.policy.predict(obs)
                    obs, reward, done, truncated, info = env.step(action)
                    cumulative_reward += gamma * reward
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