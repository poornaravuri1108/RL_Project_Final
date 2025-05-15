import gymnasium as gym
import torch
import os
import numpy as np
from gymnasium.wrappers import AtariPreprocessing, FrameStack

from train_online_ppo import ActorCritic

MODEL_PATH = "../models/online_ppo_pong_final.pt"
VIDEO_DIR = "videos/ppo/"

os.makedirs(VIDEO_DIR, exist_ok=True)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    env = gym.make("ALE/Pong-v5", render_mode="rgb_array", frameskip=1)
    env = gym.wrappers.RecordVideo(env, video_folder=VIDEO_DIR, episode_trigger=lambda x: True)
    
    env = AtariPreprocessing(
        env, 
        frame_skip=4,
        grayscale_obs=True,
        grayscale_newaxis=False,
        scale_obs=False,
        screen_size=84
    )
    env = FrameStack(env, 4)
    
    input_channels = 4  
    model = ActorCritic(input_channels, env.action_space.n).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    obs, _ = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        obs_np = np.array(obs)
        obs_tensor = torch.tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
        
        with torch.no_grad():
            logits, _ = model(obs_tensor)
            action = torch.argmax(logits, dim=1)
            
        obs, reward, terminated, truncated, info = env.step(action.item())
        total_reward += reward
        done = terminated or truncated

    env.close()
    print(f"Episode finished with reward: {total_reward}")
    print(f"Video saved in {VIDEO_DIR}")

if __name__ == "__main__":
    main()
