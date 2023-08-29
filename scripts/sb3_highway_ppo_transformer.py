import torch.multiprocessing as mp
import functools
import gymnasium as gym
import seaborn as sns
from stable_baselines3 import PPO
import torch
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import os
from enum import Enum
import json
import wandb
from datetime import datetime
from torch import FloatTensor
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler, Subset
import shutil
from models.gail import GAIL
from generate_expert_data import expert_data_collector, retrieve_agent, extract_post_processed_expert_data
from forward_simulation import make_configure_env, append_key_to_dict_of_dict, simulate_with_model
from sb3_callbacks import CustomCheckpointCallback, CustomMetricsCallback, CustomCurriculamCallback
from utilities import  write_module_hierarchy_to_file, DefaultActorCriticPolicy, CustomDataset, CustomExtractor, clear_and_makedirs
import warnings
from imitation.algorithms import bc
from python_config import sweep_config, env_kwargs
import matplotlib.pyplot as plt
import zipfile
from imitation.algorithms import bc
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
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

def save_checkpoint(project, run_name, epoch, trainer, metrics_plot_path):

    with wandb.init(
                        project=project, 
                        magic=True,
                    ) as run:
                    # if epoch is None:
                    epoch = "final"
                    run.log({f"metrics_plot": wandb.Image(metrics_plot_path)})
                    run.name = run_name
                    # Log the model as an artifact in wandb
                    torch.save(trainer , f"models_archive/BC_agent_{epoch}.pth") 
                    artifact = wandb.Artifact("trained_model_directory", type="model_directory")
                    artifact.add_dir("models_archive")
                    run.log_artifact(artifact)
    wandb.finish()
    clear_and_makedirs("models_archive")


class DownSamplingSampler(SubsetRandomSampler):
    def __init__(self, labels, class_weights, num_samples):
        """
        Args:
            class_weights (list): List of class weights.
            num_samples (int): Total number of samples to keep.
            seed (int): Seed for the random number generator.
        """
        self.class_weights = class_weights
        self.num_samples = num_samples
        # self.generator = torch.Generator()
        # self.generator.manual_seed(seed)
        self.class_labels = labels
        self.unique_labels = np.unique(self.class_labels)
        self.num_samples_per_class = int(num_samples/ len(self.unique_labels))
        self.indices = self._select_samples()

    def _select_samples(self):
        # Calculate the downsampled indices for each class
        self.downsampled_indices = []
        for class_label in self.unique_labels:
            class_indices = np.where(self.class_labels == class_label)[0]
            max_samples = min(len(class_indices), self.num_samples_per_class)
            downsampled_indices = class_indices[:max_samples]
            self.downsampled_indices.append(downsampled_indices)

        # Combine the downsampled indices for all classes
        return np.concatenate(self.downsampled_indices)
    
    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

def create_dataloaders(zip_filename, extract_path, device, **kwargs):
    # Extract the HDF5 files from the zip archive
    # These files may be alredy existing because of a previous post process step.
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

    # Create separate datasets for each HDF5 file
    train_datasets = [CustomDataset(hdf5_name, device) for hdf5_name in hdf5_train_file_names]

    # Create a combined dataset from the individual datasets
    combined_train_dataset = ConcatDataset(train_datasets)
    # Create shuffled indices for the combined dataset
    shuffled_indices = np.arange(len(combined_train_dataset))
    np.random.shuffle(shuffled_indices)

    # Create a shuffled version of the combined dataset using Subset
    shuffled_combined_train_dataset = Subset(combined_train_dataset, shuffled_indices)

    # Calculate the class frequencies
    all_actions = [sample['acts'] for sample in combined_train_dataset]
    action_frequencies = np.bincount(all_actions)
    class_weights = 1.0 / np.sqrt(action_frequencies)
    # class_weights =  np.array([np.exp(-freq/action_frequencies.sum()) for freq in action_frequencies])
    class_weights = class_weights / class_weights.sum()
    print(" class_weights at the end ", class_weights, " action_frequencies ", action_frequencies)

    # Calculate the least represented count
    least_represented_count = np.min(action_frequencies)

    # Get the number of unique action types
    num_action_types = len(np.unique(all_actions))

    num_samples=int(least_represented_count * num_action_types )
    desired_num_samples = 10000  # Adjust this value as needed
    seed = 42
    sampler = DownSamplingSampler(
                                    labels = all_actions,
                                    class_weights = class_weights, 
                                    num_samples= num_samples
                                 )
    print(" class_weights ", class_weights, " num_samples ", num_samples, " original samples fraction ", num_samples/len(all_actions))
    train_data_loader = DataLoader(
                                        shuffled_combined_train_dataset, 
                                        batch_size=kwargs['batch_size'], 
                                        # shuffle=True,
                                        sampler=sampler,
                                        drop_last=True,
                                        num_workers=n_cpu,
                                        # pin_memory=True,
                                        # pin_memory_device=device,
                                 ) 
    return train_data_loader, hdf5_train_file_names, hdf5_val_file_names


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

    # # Calculate evaluation metrics for training
    # tr_accuracy = accuracy_score(true_labels, predicted_labels)
    # tr_precision = precision_score(true_labels, predicted_labels, average=None)
    # tr_recall = recall_score(true_labels, predicted_labels, average=None)
    # tr_f1 = f1_score(true_labels, predicted_labels, average=None)



    # print("--------  Training data metrics for reference---------------")
    # print("Accuracy:", accuracy, np.mean(tr_accuracy))
    # print("Precision:", precision,  np.mean(tr_precision))
    # print("Recall:", recall, np.mean(tr_recall))
    # print("F1 Score:", f1, np.mean(tr_f1))


    plt.figure(figsize=(8, 6))
    class_labels = [ ACTIONS_ALL[idx] for idx in range(len(ACTIONS_ALL))]
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.savefig(training_kwargs['plot_path'])
    # plt.show()  
    print("saved confusion matrix")
    return accuracy, precision, recall, f1


if __name__ == "__main__":
    
    DAGGER = True
    policy_kwargs = dict(
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
        expert_data_collector(  
                                oracle_agent,
                                data_folder_path = extract_path,
                                zip_filename=zip_filename,
                                total_iterations = 100,
                                **{**env_kwargs, **{'expert':None}}           
                             )
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

        # expert_data_file = "expert_train_data_0.h5"


        # exp_obs, exp_acts, _ = extract_post_processed_expert_data(expert_data_file)
        # exp_obs = FloatTensor(exp_obs)
        # exp_acts = FloatTensor(exp_acts)

        def train_gail_agent(
                                # exp_obs=exp_obs, 
                                # exp_acts=exp_acts,
                                zip_filename, 
                                gail_agent_path = None, 
                                env_kwargs = None,
                                train_config = None,
                                ** training_kwargs
                            ):
            train_data_loaders, hdf5_train_file_names, hdf5_val_file_names = create_dataloaders(
                                                                                                    zip_filename,
                                                                                                    extract_path, 
                                                                                                    device=training_kwargs['device'],
                                                                                                    batch_size=training_kwargs['batch_size']
                                                                                                )
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

        def train_sweep(data_loaders, config=None):
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
                                                                                    # exp_obs=exp_obs, 
                                                                                    # exp_acts=exp_acts,
                                                                                    zip_filename ,
                                                                                    gail_agent_path=gail_agent_path, 
                                                                                    env_kwargs = env_kwargs,
                                                                                    train_config = train_config,
                                                                                    device=device,
                                                                                    batch_size=batch_size
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
                clear_and_makedirs("models_archive")

                shutil.move("optimal_gail_agent.pth", "models_archive/optimal_gail_agent.pth")
                shutil.move("final_gail_agent.pth", "models_archive/final_gail_agent.pth")

                # Log the model as an artifact in wandb
                artifact = wandb.Artifact("trained_model_directory", type="model_directory")
                artifact.add_dir("models_archive")
                run.log_artifact(artifact)

                    
        # rewards, optimal_agent = train_gail_agent(exp_obs=exp_obs, exp_acts=exp_acts, **env_kwargs)

        wandb.agent(
                     sweep_id=sweep_id, 
                     function=lambda: train_sweep(data_loaders)
                   )
        wandb.finish()
    elif train == TrainEnum.IRLDEPLOY:
        append_key_to_dict_of_dict(env_kwargs,'config','duration',40)
        append_key_to_dict_of_dict(env_kwargs,'config','vehicles_count',150)
        env_kwargs.update({'reward_oracle':None})
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
        rng=np.random.default_rng()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        project = "BC_1"
        # device = 'cpu'
        policy = DefaultActorCriticPolicy(env, device)
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
                train_data_loader, hdf5_train_file_names, hdf5_val_file_names = create_dataloaders(
                                                                                                      zip_filename,
                                                                                                      extract_path, 
                                                                                                      device=device,
                                                                                                      batch_size=training_kwargs['batch_size']
                                                                                                  )
                
                if False:                                                                                   
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
                    plt.show()

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

                last_epoch = (epoch ==num_epochs-1)
                num_mini_batches = 150000 if last_epoch else 25000 # Mini epoch here correspond to typical epoch
                trainer.set_demonstrations(train_data_loader)
                trainer.train(n_batches=num_mini_batches)  
                if not last_epoch and DAGGER:
                    expert_data_collector(
                                            trainer.policy, # This is the exploration policy
                                            data_folder_path = extract_path,
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
                                                                                hdf5_train_file_names, 
                                                                                hdf5_val_file_names, 
                                                                                plot_path=f"{extract_path}/heatmap_{epoch}.png" 
                                                                              )
                _validation_metrics["accuracy"].append(accuracy)
                _validation_metrics["precision"].append(precision)
                _validation_metrics["recall"].append(recall)
                _validation_metrics["f1"].append(f1)
            epochs = range(num_epochs)


            accuracy, precision, recall, f1 = calculate_validation_metrics(
                                                                            trainer, 
                                                                            hdf5_train_file_names, 
                                                                            hdf5_val_file_names, 
                                                                            plot_path=f"{extract_path}/heatmap_{epoch}.png" 
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
        append_key_to_dict_of_dict(env_kwargs,'config','vehicles_count',150)
        append_key_to_dict_of_dict(env_kwargs,'config','real_time_rendering',True)
        env = make_configure_env(**env_kwargs)
        BC_agent                            = retrieve_agent(
                                                            artifact_version='trained_model_directory:latest',
                                                            agent_model = 'BC_agent_5.pth',
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
                    print("speed: ",env.vehicle.speed," ,reward: ", reward, " ,cumulative_reward: ",cumulative_reward)
                    env.render()
                print("--------------------------------------------------------------------------------------")

