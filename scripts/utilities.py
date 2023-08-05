
import torch.nn as nn
import h5py
import sys

    
def extract_expert_data(filename):
    exp_obs  = []
    exp_acts = []
    with h5py.File(filename, 'r') as hf:
        # List all the episode groups in the HDF5 file
        episode_groups = list(hf.keys())

        # Iterate through each episode group
        for episode_name in episode_groups:
            episode = hf[episode_name]

            # List all datasets (exp_obs and exp_acts) in the episode group
            datasets = list(episode.keys())

            # Iterate through each dataset in the episode group
            for dataset_name in datasets:
                dataset = episode[dataset_name]

                # Append the data to the corresponding list
                if dataset_name.startswith('exp_obs'):
                    exp_obs.append(dataset[:])
                elif dataset_name.startswith('exp_acts'):
                    exp_acts.append(dataset[()]) 

    return  exp_obs, exp_acts


def write_module_hierarchy_to_file(model, file):
    def write_module_recursive(module, file=None, indent='', processed_submodules=None):
        if file is None:
            file = sys.stdout
        if processed_submodules is None:
            processed_submodules = set()

        num_members = [tuple(_.shape) for _ in module.parameters()]
        # num_members = len(list(module.modules())) - 1
        module_name = f'{module.__class__.__name__} (ID: {id(module)})'
        file.write(f'{indent}├─{module_name} '+ ' containing '+ str(len(num_members))  + ' items\n')

        if isinstance(module, nn.Sequential):
            for submodule in module:
                write_module_recursive(submodule, file, indent + '    ')
        elif isinstance(module, nn.ModuleList):
            for idx, submodule in enumerate(module):
                file.write(f'{indent}    ├─ModuleList[{idx}]\n')
                write_module_recursive(submodule, file, indent + '        ')
        else:
            for name, submodule in module._modules.items():
                if submodule not in processed_submodules:
                    processed_submodules.add(submodule)
                    write_module_recursive(submodule, file, indent + '    ')

            for name, submodule in module._parameters.items():
                if submodule is not None:
                    if submodule not in processed_submodules:
                        processed_submodules.add(submodule)
                        file.write(f'{indent}    ├─{name}: {submodule.shape}\n')

    write_module_recursive(model, file, processed_submodules=set())
