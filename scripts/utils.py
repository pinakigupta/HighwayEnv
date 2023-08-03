
from pyvirtualdisplay import Display
from gymnasium.wrappers import RecordVideo
from pathlib import Path
import base64
import torch.nn as nn
import h5py


display = Display(visible=0, size=(1400, 900))
display.start()


def record_videos(env, video_folder="videos"):
    wrapped = RecordVideo(env, video_folder=video_folder, episode_trigger=lambda e: True)

    # Capture intermediate frames
    env.unwrapped.set_record_video_wrapper(wrapped)

    return wrapped


def show_videos(path="videos"):
    from IPython import display as ipythondisplay
    html = []
    for mp4 in Path(path).glob("*.mp4"):
        video_b64 = base64.b64encode(mp4.read_bytes())
        html.append('''<video alt="{}" autoplay
                      loop controls style="height: 400px;">
                      <source src="data:video/mp4;base64,{}" type="video/mp4" />
                 </video>'''.format(mp4, video_b64.decode('ascii')))
    ipythondisplay.display(ipythondisplay.HTML(data="<br>".join(html)))
    
    
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


