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
gym_alias = os.getenv("GYM_ALIAS", "gymnasium")
import importlib
gym = importlib.import_module(gym_alias)
import json
import wandb
import torchvision.transforms.functional as TF
from datetime import datetime
import time
from models.gail import GAIL
from generate_expert_data import expert_data_collector, retrieve_agent
from forward_simulation import make_configure_env, append_key_to_dict_of_dict, simulate_with_model
from sb3_callbacks import CustomCheckpointCallback, CustomMetricsCallback, CustomCurriculamCallback
from utilities import *
from utils import record_videos
import warnings
from imitation.algorithms import bc
from python_config import *
import matplotlib.pyplot as plt
from imitation.algorithms import bc
import importlib
import pandas as pd
import tracemalloc
from highway_env.envs.common.observation import *
warnings.filterwarnings("ignore")

from highway_env.envs.common.action import DiscreteMetaAction
ACTIONS_ALL = DiscreteMetaAction.ACTIONS_ALL




def timenow():
    return now.strftime("%H%M")


def print_stack_size():
    while True:
        current_size, peak_size = tracemalloc.get_traced_memory()
        print(f"Current memory size: {current_size / (1024 * 1024):.2f} MB")
        time.sleep(15)  # Adjust the interval as needed
# ==================================
#        Main script  20 
# ==================================


import threading
if __name__ == "__main__":
    torch.cuda.empty_cache()
    TRACE = False
    if TRACE:
        tracemalloc.start()  # Start memory tracing
        # Create a multiprocessing process to periodically print stack size
        stack_size_thread = threading.Thread(target=print_stack_size)
        stack_size_thread.daemon = True
        stack_size_thread.start()

    DAGGER = True
    
    if env_kwargs['config']['observation'] == env_kwargs['config']['KinematicObservation']:
        features_extractor_class=CombinedFeatureExtractor
        features_extractor_kwargs=attention_network_kwargs

        if features_extractor_class is  CombinedFeatureExtractor:
            action_extractor_kwargs = dict(
            action_extractor_kwargs = {
                                "feature_size": 8, 
                                "dropout_factor": 0.0, 
                                "obs_space": make_configure_env(**env_kwargs).env.observation_space,
                                "act_space": make_configure_env(**env_kwargs).env.action_space
                           })
            features_extractor_kwargs = {**features_extractor_kwargs, **action_extractor_kwargs}
        policy_kwargs = dict(
                # policy=MLPPolicy,
                features_extractor_class=features_extractor_class,
                features_extractor_kwargs=features_extractor_kwargs,
            )
        
        append_key_to_dict_of_dict(env_kwargs,'config','screen_width',960*3)
        append_key_to_dict_of_dict(env_kwargs,'config','screen_height',180*2)
    else:
        policy_kwargs = dict(
                                features_extractor_class=CustomVideoFeatureExtractor,
                                # features_extractor_class=CustomImageExtractor,
                                features_extractor_kwargs=dict(hidden_dim=64, augment_image=True),
                            )


    WARM_START = False
    # Get the current date and time
    now = datetime.now()
    month = now.strftime("%m")
    day = now.strftime("%d")

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
    minibatch_size = batch_size
    num_deploy_rollouts = 5

    try:
        project= project_names[train.value]                           
        if   train == TrainEnum.EXPERT_DATA_COLLECTION: # EXPERT_DATA_COLLECTION
            append_key_to_dict_of_dict(env_kwargs,'config','mode','MDPVehicle')
            append_key_to_dict_of_dict(env_kwargs,'config','deploy',True)
            policy = None
            env = make_configure_env(**env_kwargs)
            if policy:
                # oracle_agent                            = retrieve_agent(
                #                                                             artifact_version='trained_model_directory:latest',
                #                                                             agent_model = 'agent_final.pt',
                #                                                             device=device,
                #                                                             project=project
                #                                                         )
                # policy = DefaultActorCriticPolicy(make_configure_env(**env_kwargs), device, **policy_kwargs)
                policy = RandomPolicy(env=env, device=device, **policy_kwargs)
                # policy.load_state_dict(oracle_agent.state_dict())
                policy.eval()
                print('EXPERT_DATA_COLLECTION using PREVIOUS POLICY for exploration')
            else:
                print('EXPERT_DATA_COLLECTION using IDM+MOBIL for exploration')
            
            
            expert_data_collector(  
                                    policy,
                                    extract_path = extract_path,
                                    zip_filename=zip_filename,
                                    delta_iterations = 10,
                                    **{**env_kwargs, **{'expert':'MDPVehicle'}}           
                                )
            print(" finished collecting data for ALL THE files ")
        elif train == TrainEnum.RLTRAIN: # training  # Reinforcement learning with curriculam update 
            env_kwargs.update({'reward_oracle':None})
            append_key_to_dict_of_dict(env_kwargs,'config','duration',10)
            append_key_to_dict_of_dict(env_kwargs,'config','EGO_LENGTH',8)
            append_key_to_dict_of_dict(env_kwargs,'config','EGO_WIDTH',4)
            append_key_to_dict_of_dict(env_kwargs,'config','max_vehicles_count',80)
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
                                                                                                    type = 'train',
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
                                    delta_iterations = 2,
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
            #                                                     project=project
            #                                                     )
            
            wandb.init(project=project, name="inference")
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
                                                                extractor=model.features_extractor.extractor,
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
                    end_time = time.time()
                print("speed: ",env.vehicle.speed," ,reward: ", reward, " ,cumulative_reward: ",cumulative_reward)
                print("--------------------------------------------------------------------------------------")
        elif train == TrainEnum.BC:
            env_kwargs.update({'reward_oracle':None})
            append_key_to_dict_of_dict(env_kwargs,'config','deploy',True)
            # env = make_configure_env(**env_kwargs)
            # env=env.unwrapped
            # env = make_vec_env(
            #                     make_configure_env, 
            #                     n_envs=n_cpu, 
            #                     vec_env_cls=SubprocVecEnv, 
            #                     env_kwargs=env_kwargs
            #                 )
            env = make_configure_env(**env_kwargs)
            # state_dim = env.observation_space.high.shape[0]*env.observation_space.high.shape[1]
            rng=np.random.default_rng()
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
                                        minibatch_size=kwargs['minibatch_size'],
                                        device = device,
                                        l2_weight = 0.0001,
                                        # ent_weight= 0.01,
                                        policy=policy
                                    )        
        

            def _train(env, policy, zip_filename, extract_path, device=device, **training_kwargs):
                num_epochs = training_kwargs['num_epochs'] # These are dagger epochs
                checkpoint_interval = num_epochs//2
                append_key_to_dict_of_dict(env_kwargs,'config','mode','MDPVehicle')
                trainer = create_trainer(env, policy, batch_size=batch_size, minibatch_size=minibatch_size, num_epochs=num_epochs, device=device) # Unfotunately needed to instantiate repetitively
                print(" trainer policy (train_mode ?)", trainer.policy.training)
                epoch = None
                train_datasets = []                    
                visited_data_files = set([])
                metricses = {}
                for epoch in range(num_epochs): # Epochs here correspond to new data distribution (as maybe collecgted through DAGGER)
                    print(f'Loadng training data loader for epoch {epoch}')
                    train_data_loader                                            = create_dataloaders(
                                                                                                          zip_filename,
                                                                                                          train_datasets, 
                                                                                                          type = 'train',
                                                                                                          device=device,
                                                                                                          batch_size=minibatch_size,
                                                                                                          n_cpu = n_cpu,
                                                                                                          visited_data_files=visited_data_files
                                                                                                      )
                    # train_data_loader = CustomDataLoader(
                    #                                         zip_filename, 
                    #                                         device, 
                    #                                         visited_data_files, 
                    #                                         batch_size = minibatch_size, 
                    #                                         n_cpu=n_cpu, 
                    #                                         chunk_size=15000,
                    #                                         type='train'
                    #                                     )
                    print(f'Loaded training data loader for epoch {epoch}')
                    last_epoch = (epoch ==num_epochs-1)
                    num_mini_batches = 55600 if last_epoch else 7500*(1+epoch) # Mini epoch here correspond to typical epoch
                    TrainPartiallyPreTrained = (env_kwargs['config']['observation'] == env_kwargs['config']['GrayscaleObservation'])
                    if TrainPartiallyPreTrained: 
                        trainer.policy.features_extractor.set_grad_video_feature_extractor(requires_grad=False)
                    trainer.set_demonstrations(train_data_loader)
                    print(f'Beginning Training for epoch {epoch}')
                    trainer.train(
                                    n_batches=num_mini_batches,
                                    # log_rollouts_venv = env,
                                    # log_rollouts_n_episodes =10,
                                 )
                    if TrainPartiallyPreTrained:
                        trainer.policy.features_extractor.set_grad_video_feature_extractor(requires_grad=True)
                        trainer.train(n_batches=25000)                   
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

                    # if checkpoint_interval !=0 and epoch % checkpoint_interval == 0 and not last_epoch:
                    #     print(f"saving check point for epoch {epoch}", epoch)
                    #     torch.save(policy , f"models_archive/BC_agent_{epoch}.pth")
                    if metrics_plot_path:
                        print(f'Beginning validation for epoch {epoch}')
                        metrics                      = calculate_validation_metrics(
                                                                                        policy, 
                                                                                        zip_filename=zip_filename,
                                                                                        device=torch.devie('cpu'),
                                                                                        batch_size=batch_size,
                                                                                        n_cpu=n_cpu,
                                                                                        val_batch_count=500,
                                                                                        chunk_size=500,
                                                                                        type='val',
                                                                                        plot_path=None
                                                                                    )
                        {key: metricses.setdefault(key, []).append(value) for key, value in metrics.items()}
                        print(f'End validation for epoch {epoch}')
                    policy.to(device)
                    policy.train()
                



                if metrics_plot_path and metricses:
                    # Plotting metrics
                    plt.figure(figsize=(10, 6))
                    epochs = range(num_epochs)
                    for metric_name, metric_values in metricses.items():
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
                                                                    minibatch_size=minibatch_size
                                                                )
            final_policy = bc_trainer.policy
            val_device = torch.device('cpu')
            final_policy.to(val_device)
            final_policy.eval()
            if True:
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
            # train_datasets =[]
            # data_loader              = CustomDataLoader(
            #                                                 zip_filename, 
            #                                                 device=val_device,
            #                                                 batch_size=batch_size,
            #                                                 n_cpu=n_cpu,
            #                                                 val_batch_count=500,
            #                                                 chunk_size=500,
            #                                                 type='val',
            #                                                 plot_path=None,
            #                                                 visited_data_files = set([])
            #                                             ) 
            # metrics                         = calculate_validation_metrics(
            #                                                                 data_loader,
            #                                                                 final_policy, 
            #                                                                 zip_filename=zip_filename,
            #                                                                 device=val_device,
            #                                                                 batch_size=batch_size,
            #                                                                 n_cpu=n_cpu,
            #                                                                 val_batch_count=500,
            #                                                                 chunk_size=500,
            #                                                                 type='val',
            #                                                                 plot_path=None
            #                                                               )
            # metrics                        = calculate_validation_metrics(
            #                                                                 data_loader,
            #                                                                 final_policy, 
            #                                                                 zip_filename=zip_filename,
            #                                                                 device=val_device,
            #                                                                 batch_size=batch_size,
            #                                                                 n_cpu=n_cpu,
            #                                                                 val_batch_count=500,
            #                                                                 chunk_size=500,
            #                                                                 type='train',
            #                                                                 plot_path=None
            #                                                               )
            print('Ending final validation step and plotting the heatmap ')
            final_policy.to(device)
        elif train == TrainEnum.BCDEPLOY:
            env_kwargs.update({'reward_oracle':None})
            # env_kwargs.update({'render_mode': 'human'})
            append_key_to_dict_of_dict(env_kwargs,'config','max_vehicles_count',125)
            append_key_to_dict_of_dict(env_kwargs,'config','min_lanes_count',2)
            append_key_to_dict_of_dict(env_kwargs,'config','real_time_rendering',True)
            append_key_to_dict_of_dict(env_kwargs,'config','deploy',True)
            append_key_to_dict_of_dict(env_kwargs,'config','duration',80)
            append_key_to_dict_of_dict(env_kwargs,'config','offscreen_rendering',False)
            env = make_configure_env(**env_kwargs)
            env = record_videos(env=env, name_prefix = 'BC', video_folder='videos/BC')
            # BC_agent                            = retrieve_agent(
            #                                                         artifact_version='trained_model_directory:latest',
            #                                                         agent_model = 'agent_final.pt',
            #                                                         device=device,
            #                                                         project=project
            #                                                     )
            BC_agent                            = retrieve_agent(
                                                        artifact_version='trained_model_directory:latest',
                                                        agent_model = 'agent_final.pth',
                                                        device=device,
                                                        project=project
                                                        )
            gamma = 1.0
            env.render()
            # policy = DefaultActorCriticPolicy(env, device, **policy_kwargs)
            # policy.load_state_dict(BC_agent.state_dict())
            policy = BC_agent
            policy.to(device)
            policy.eval()
            if isinstance(env.observation_type, KinematicObservation):
                env.viewer.set_agent_display(
                                                functools.partial(
                                                                    display_vehicles_attention, 
                                                                    env=env, 
                                                                    extractor=policy.features_extractor.obs_extractor.extractor,
                                                                    device=device
                                                                )
                                             )
                image_space_obs = False
            else:
                image_space_obs = True

            
            if image_space_obs:   
                fig = plt.figure(figsize=(8, 16))
                import cv2
            predicted_labels = []
            true_labels = []
            for _ in range(num_deploy_rollouts):
                obs, info = env.reset()
                env.step({'long':4, 'lat':1})
                done = truncated = False
                cumulative_reward = 0
                while not (done or truncated):
                    start_time = time.time()
                    expert_action , _= env.vehicle.discrete_action()
                    expert_action = [env.action_type.actions_indexes[key][expert_action[key]] for key in ['long', 'lat']] 
                    true_labels.append(expert_action)
                    action, _ = policy.predict(obs)
                    # action_logits = torch.nn.functional.softmax(policy.action_net(policy.mlp_extractor(policy.features_extractor(torch.tensor(obs).to(device)))[0]))
                    # action = np.array(torch.argmax(action_logits, dim=-1).item())
                    predicted_labels.append(action)
                    # env.vehicle.actions = []
                    obs, reward, done, truncated, info = env.step(action)
                    obs[-2] = 0
                    end_time = time.time()
                    cumulative_reward += gamma * reward
                    if image_space_obs:
                        height, width = env.observation_space.shape[1], env.observation_space.shape[2]
                        observations = torch.tensor(obs, dtype=torch.float32)
                        observations = observations.view( -1, 1, height, width)
                        observations = torch.cat([observations, observations, observations], dim=1)
                        transformed_obs = policy.features_extractor.video_preprocessor(observations)
                        for i in range(3,4):
                            raw_image = obs[i,:]
                            image =  transformed_obs[0, i,:].cpu().numpy()
                            # action_logits = policy.action_net(input_tensor)
                            # action_logits = policy.action_net(policy.mlp_extractor(policy.features_extractor(torch.Tensor(obs).unsqueeze(0)))[0])
                            # selected_action = torch.argmax(action_logits, dim=1)
                            # action_logits[0, selected_action].backward()
                            # gradients = input_tensor.grad.squeeze()
                            # # Normalize and overlay the gradient-based heatmap on the original image
                            # heatmap = gradients.numpy()
                            # heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))  # Normalize between 0 and 1
                            # heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
                            # heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
                            # result  = cv2.addWeighted(image.astype(np.uint8), 0.5, heatmap, 0.5, 0)

                            ax1 = plt.subplot(121)  # 2 rows, 1 column, subplot 1
                            ax1.imshow(raw_image, cmap='gray', origin='lower', aspect=1.0)
                            ax1.set_xlim(20, 40)

                            ax2 = plt.subplot(122)  # 2 rows, 1 column, subplot 2
                            ax2.imshow(image, cmap='gray', origin='lower', aspect=1.0)
                            ax2.set_xlim(20, 80)   

                            plt.show(block=False)
                            plt.pause(0.01)
                    # env.render()
                    frequency = 1/(end_time-start_time)
                    print(f"Execution frequency is {frequency}")
                print("speed: ",env.vehicle.speed," ,reward: ", reward, " ,cumulative_reward: ",cumulative_reward)
                print("--------------------------------------------------------------------------------------")
            true_labels = true_labels @ label_weights
            # predicted_labels = predicted_labels @ label_weights
            accuracy = accuracy_score(true_labels, predicted_labels)
            precision = precision_score(true_labels, predicted_labels, average=None)
            recall    = recall_score(true_labels, predicted_labels, average=None)
            f1        = f1_score(true_labels, predicted_labels, average=None)
            print("Accuracy:", accuracy, np.mean(accuracy))
            print("Precision:", precision, np.mean(precision))
            print("Recall:", recall, np.mean(recall))
            print("F1 Score:", f1, np.mean(f1))
        elif train == TrainEnum.ANALYSIS:
            val_batch_count = 500000
            train_data_loader                                             = create_dataloaders(
                                                                                                          zip_filename,
                                                                                                          train_datasets=[], 
                                                                                                          type = 'train',
                                                                                                          device=device,
                                                                                                          batch_size=minibatch_size,
                                                                                                          n_cpu = n_cpu,
                                                                                                          visited_data_files=set([])
                                                                                                      )
            # Create a DataFrame from the data loader
            data_list = np.empty((0, 100))
            actions = []
            with torch.no_grad():
                for batch_number, batch in enumerate(train_data_loader):
                    if batch_number >= val_batch_count:
                        break
                    whole_batch_states = batch['obs'].reshape(-1, 100) 
                    actions.extend(batch['acts'].cpu().numpy().astype(int))
                    data_list = np.vstack((data_list, whole_batch_states.cpu().numpy()))
            data_df = pd.DataFrame(data_list)
            actions = np.array(actions)
            action_counts = np.bincount(actions.astype(int))
            print('action_counts ', action_counts)

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
            print('normalized_counts', normalized_counts)

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
        elif train == TrainEnum.VALIDATION:
            policy                            = retrieve_agent(
                                                                artifact_version='trained_model_directory:latest',
                                                                agent_model = 'agent_final.pth',
                                                                device=device,
                                                                project=project
                                                              )
            val_device = torch.device('cpu')
            policy.to(val_device)
            policy.eval()
            type = 'val'
            val_data_loader                                             =  CustomDataLoader(
                                                                                            zip_filename, 
                                                                                            device=val_device,
                                                                                            batch_size=batch_size,
                                                                                            n_cpu=n_cpu,
                                                                                            val_batch_count=5000,
                                                                                            chunk_size=500,
                                                                                            type= type,
                                                                                            plot_path=None,
                                                                                            visited_data_files = set([])
                                                                                        ) 
            metrics                      = calculate_validation_metrics(
                                                                            val_data_loader,
                                                                            policy, 
                                                                            zip_filename=zip_filename,
                                                                            device=val_device,
                                                                            batch_size=batch_size,
                                                                            n_cpu=n_cpu,
                                                                            val_batch_count=5000,
                                                                            chunk_size=500,
                                                                            type= type,
                                                                            validation = True,
                                                                            plot_path=None
                                                                          )
            type = 'train'
            train_data_loader = CustomDataLoader(
                                                zip_filename, 
                                                device=val_device,
                                                batch_size=batch_size,
                                                n_cpu=n_cpu,
                                                val_batch_count=500,
                                                chunk_size=500,
                                                type= type,
                                                plot_path=None,
                                                visited_data_files = set([])
                                              ) 
            metrics                      = calculate_validation_metrics(
                                                                            train_data_loader,
                                                                            policy, 
                                                                            zip_filename=zip_filename,
                                                                            device=val_device,
                                                                            batch_size=batch_size,
                                                                            n_cpu=n_cpu,
                                                                            val_batch_count=500,
                                                                            chunk_size=500,
                                                                            type= type,
                                                                            validation = True,
                                                                            plot_path=None
                                                                          )
    except KeyboardInterrupt:
        if TRACE:
            tracemalloc.stop()  # Stop memory tracing when done
        # Save a reference to the original stdout
        original_stdout = sys.stdout
        try:
            sys.stdout = open(os.devnull, 'w')
            raise KeyboardInterrupt("Manually raised KeyboardInterrupt")
        except KeyboardInterrupt:
            # Use a lambda function to temporarily set sys.excepthook to suppress output
             sys.excepthook = lambda exctype, value, traceback: None
             raise
        finally:
            # Restore the original stdout
            sys.stdout = original_stdout