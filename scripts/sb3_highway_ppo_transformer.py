import functools
import gymnasium as gym
import seaborn as sns
from stable_baselines3 import PPO
import torch
import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import os
import multiprocessing
from enum import Enum
import json
import wandb
from datetime import datetime
from torch import FloatTensor
from torch.utils.data import DataLoader
import shutil
from models.gail import GAIL
from generate_expert_data import collect_expert_data, postprocess
from forward_simulation import make_configure_env, append_key_to_dict_of_dict, simulate_with_model
from sb3_callbacks import CustomCheckpointCallback, CustomMetricsCallback, CustomCurriculamCallback
from utilities import extract_post_processed_expert_data, write_module_hierarchy_to_file, DefaultActorCriticPolicy, CustomDataset, CustomExtractor, retrieve_gail_agents
import warnings
from imitation.algorithms import bc
from python_config import sweep_config, env_kwargs
import matplotlib.pyplot as plt
import zipfile
from imitation.algorithms import bc
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

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

train = TrainEnum.BC



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
    policy_kwargs = dict(
            features_extractor_class=CustomExtractor,
            features_extractor_kwargs=attention_network_kwargs,
        )

    if False:
        optimal_gail_agent, final_gail_agent = retrieve_gail_agents(
                                                                    # env= make_configure_env(**env_kwargs).unwrapped, # need only certain parameters
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
    validation_data_file = 'expert_data_rel_val.h5'
    zip_filename = 'expert_data.zip'
    n_cpu =  multiprocessing.cpu_count()
    device = torch.device("cpu")
    extract_path = "data/expert_data"


    

    
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
        append_key_to_dict_of_dict(env_kwargs,'config','duration',20)

        total_iterations = 1000  # The total number of loop iterations
        with zipfile.ZipFile(zip_filename, 'w') as zipf:
            # Create an outer tqdm progress bar
            outer_bar = tqdm(total=total_iterations, desc="Outer Loop Progress")
            for filenum in range(total_iterations):
                collect_expert_data  (
                                            model,
                                            config["num_expert_steps"],
                                            train_filename=expert_data_file,
                                            validation_filename=validation_data_file,
                                            **env_kwargs
                                      )
                # print("collect data complete")
                exp_file = f'data/expert_train_data_{filenum}.h5'
                val_file = f'data/expert_val_data_{filenum}.h5'
                postprocess(expert_data_file, exp_file)
                postprocess(validation_data_file, val_file)
                zipf.write(exp_file)
                zipf.write(val_file)
                outer_bar.update(1)
            outer_bar.close()
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

        exp_obs, exp_acts, _ = extract_post_processed_expert_data(expert_data_file)
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
        append_key_to_dict_of_dict(env_kwargs,'config','vehicles_count',150)
        env_kwargs.update({'reward_oracle':None})
        # env_kwargs.update({'render_mode': None})
        env = make_configure_env(**env_kwargs)
        optimal_gail_agent, final_gail_agent = retrieve_gail_agents(
                                                                    # env=env,
                                                                    artifact_version='trained_model_directory:latest',
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
        append_key_to_dict_of_dict(env_kwargs,'config','vehicles_count',150)
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

        num_epochs = 3
        rng=np.random.default_rng()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        policy = DefaultActorCriticPolicy(env, device)
        print("Default policy initialized ")
        
        batch_size= 128
        def create_trainer(env, policy, **kwargs):
            return       bc.BC(
                                observation_space=env.observation_space,
                                action_space=env.action_space,
                                demonstrations=None, #training_transitions,
                                rng=np.random.default_rng(),
                                batch_size=kwargs['batch_size'],
                                device = device,
                                policy=policy
                                )        
        
        def create_dataloaders(zip_filename, **kwargs):
            # Extract the HDF5 files from the zip archive
            with zipfile.ZipFile(zip_filename, 'r') as archive:
                archive.extractall(extract_path)

            # Extract the names of the HDF5 files from the zip archive
            with zipfile.ZipFile(zip_filename, 'r') as archive:
                hdf5_train_file_names = [os.path.join(extract_path, name) 
                                            for name in archive.namelist() 
                                            if name.endswith('.h5') and "train" in name]
                hdf5_val_file_names = [os.path.join(extract_path, name) 
                                            for name in archive.namelist() 
                                            if name.endswith('.h5') and "val" in name]            

            hdf5_train_file_names = hdf5_train_file_names[:2]
            hdf5_val_file_names = hdf5_val_file_names[:2]
            # Create separate datasets for each HDF5 file
            train_datasets = [CustomDataset(hdf5_name, device) for hdf5_name in hdf5_train_file_names]
            # val_datasets = [CustomDataset(hdf5_name, device) for hdf5_name in hdf5_val_file_names]
            
            # custom_dataset = CustomDataset(expert_data_file, device=device)
            train_data_loaders = [DataLoader(
                                        dataset, 
                                        batch_size=kwargs['batch_size'], 
                                        shuffle=True,
                                        drop_last=True,
                                        num_workers=n_cpu,
                                        pin_memory=True
                                    ) for dataset in train_datasets]
            return train_data_loaders, hdf5_train_file_names, hdf5_val_file_names
        
        def _train(trainer, data_loaders, **training_kwargs):
            for epoch in range(training_kwargs['num_epochs']):
                for data_loader in data_loaders:
                    trainer.set_demonstrations(data_loader)
                    trainer.train(n_epochs=1) 
                    print(trainer._logger)   

        def calculate_validation_metrics(bc_trainer, hdf5_train_file_names, hdf5_val_file_names, **training_kwargs):
            true_labels = []
            predicted_labels = []
            # Iterate through the validation data and make predictions
            with torch.no_grad():
                for val_data_file in hdf5_val_file_names:
                    val_obs, val_acts, val_dones = extract_post_processed_expert_data(val_data_file)
                    predicted_labels.extend([bc_trainer.policy.predict(obs)[0] for obs in val_obs])
                    true_labels.extend(val_acts)

            # Calculate evaluation metrics
            accuracy = accuracy_score(true_labels, predicted_labels)
            precision = precision_score(true_labels, predicted_labels, average=None)
            recall = recall_score(true_labels, predicted_labels, average=None)
            f1 = f1_score(true_labels, predicted_labels, average=None)
            conf_matrix = confusion_matrix(true_labels, predicted_labels)

            # Print the metrics
            print("Accuracy:", accuracy, np.mean(accuracy))
            print("Precision:", precision, np.mean(precision))
            print("Recall:", recall, np.mean(recall))
            print("F1 Score:", f1, np.mean(f1))


            predicted_labels = []
            true_labels = []
            with torch.no_grad():
                for val_data_file in hdf5_train_file_names:
                    val_obs, val_acts, val_dones = extract_post_processed_expert_data(val_data_file)
                    predicted_labels.extend([bc_trainer.policy.predict(obs)[0] for obs in val_obs])
                    true_labels.extend(val_acts)

            # Calculate evaluation metrics
            accuracy = accuracy_score(true_labels, predicted_labels)
            precision = precision_score(true_labels, predicted_labels, average=None)
            recall = recall_score(true_labels, predicted_labels, average=None)
            f1 = f1_score(true_labels, predicted_labels, average=None)



            print("--------  Training data metrics for reference---------------")
            print("Accuracy:", accuracy, np.mean(accuracy))
            print("Precision:", precision,  np.mean(precision))
            print("Recall:", recall, np.mean(recall))
            print("F1 Score:", f1, np.mean(recall))


            plt.figure(figsize=(8, 6))
            class_labels = [ ACTIONS_ALL[idx] for idx in range(len(ACTIONS_ALL))]
            sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
            plt.xlabel("Predicted Labels")
            plt.ylabel("True Labels")
            plt.title("Confusion Matrix")
            plt.savefig(training_kwargs['plot_path'])
            # plt.show()  

        import importlib
        import python_config
        importlib.reload(python_config)
        from python_config import sweep_config
        print(sweep_config['parameters'])

        project_name = f"BC_1"
        run_name = f"sweep_{month}{day}_{timenow()}"
        sweep_id = wandb.sweep(sweep_config, project=project_name)

        def train_sweep(env, policy, config=None):
            with wandb.init(
                        project=project_name, 
                        config=config,
                        # magic=True,
                    ) as run:
                config = run.config
                print("config ", config)
                run.name = f"sweep_{month}{day}_{timenow()}"
                name = ""
                for key, value in config.items():
                    # Discard first 3 letters from the key
                    key = key[3:]

                    # Format the float value
                    value_str = "{:.2f}".format(value).rstrip('0').rstrip('.') if value and '.' in str(value) else str(value)

                    # Append the formatted key-value pair to the name
                    name += f"{key}_{value_str}_"
                plot_path = f"plot_{name}.png"
                batch_size = config.batch_size
                num_epochs = config.num_epochs 
                bc_trainer = create_trainer(env, policy, batch_size=batch_size, num_epochs=num_epochs)
                train_data_loaders, hdf5_train_file_names, hdf5_val_file_names = create_dataloaders(zip_filename, batch_size=batch_size)
                _train(bc_trainer, train_data_loaders, num_epochs=num_epochs)
                calculate_validation_metrics(bc_trainer, hdf5_train_file_names, hdf5_val_file_names, plot_path=plot_path )

                os.makedirs("models_archive", exist_ok=True)
                torch.save(bc_trainer, 'models_archive/BC_agent.pth') 

                # Log the model as an artifact in wandb
                artifact = wandb.Artifact("trained_model_directory", type="model_directory")
                artifact.add_dir("models_archive")
                run.log_artifact(artifact)
                # Log the saved plot to WandB
                run.log({f"plot_{name}": wandb.Image(plot_path)})

        wandb.agent(
                    sweep_id=sweep_id, 
                    function=lambda: train_sweep(env, policy)
                )

        wandb.finish()
            

    elif train == TrainEnum.BCDEPLOY:
        env_kwargs.update({'reward_oracle':None})
        env_kwargs.update({'render_mode': 'human'})
        append_key_to_dict_of_dict(env_kwargs,'config','vehicles_count',150)
        append_key_to_dict_of_dict(env_kwargs,'config','real_time_rendering',True)
        env = make_configure_env(**env_kwargs)
        rng=np.random.default_rng()
        wandb.init(project="BC", name="inference")
        # Access the run containing the logged artifact

        # Download the artifact
        artifact = wandb.use_artifact("trained_model:latest")
        artifact_dir = artifact.download()

        # Load the model from the downloaded artifact
        rl_agent_path = os.path.join(artifact_dir, "BC_agent.pth")
        BC_agent = torch.load(rl_agent_path, map_location=device)
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
                action, _ = BC_agent.policy.predict(obs)
                obs, reward, done, truncated, info = env.step(action)
                cumulative_reward += gamma * reward
                print("speed: ",env.vehicle.speed," ,reward: ", reward, " ,cumulative_reward: ",cumulative_reward)
                env.render()
            print("--------------------------------------------------------------------------------------")

