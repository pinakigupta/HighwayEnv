
from gymnasium.wrappers import RecordVideo
import gymnasium as gym
from pathlib import Path
import base64






def record_videos(env, name_prefix, video_folder="videos"):
    wrapped = RecordVideo(env, video_folder=video_folder, name_prefix = name_prefix, episode_trigger=lambda e: True)

    # Capture intermediate frames
    env.unwrapped.set_record_video_wrapper(wrapped) 

    return wrapped



def show_videos(path="videos"):
    from pyvirtualdisplay import Display
    display = Display(visible=0, size=(1400, 900))
    display.start()
    from IPython import display as ipythondisplay
    html = []
    for mp4 in Path(path).glob("*.mp4"):
        video_b64 = base64.b64encode(mp4.read_bytes())
        html.append('''<video alt="{}" autoplay
                      loop controls style="height: 400px;">
                      <source src="data:video/mp4;base64,{}" type="video/mp4" />
                 </video>'''.format(mp4, video_b64.decode('ascii')))
    ipythondisplay.display(ipythondisplay.HTML(data="<br>".join(html)))
    
    


