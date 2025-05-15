import gymnasium as gym
import torch
import os
import numpy as np
from gymnasium.wrappers import AtariPreprocessing, FrameStack

from train_dqn import DQNNetwork

MODEL_PATH = "../dqn_pong.pt" 
VIDEO_DIR = "videos/"

os.makedirs(VIDEO_DIR, exist_ok=True)

def select_action(model, obs, device):
    obs_np = np.array(obs)
    obs_tensor = torch.tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        q_values = model(obs_tensor)
        action = q_values.argmax(dim=1).item()
    return action

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
    
    model = DQNNetwork(4, env.action_space.n).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    obs, _ = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action = select_action(model, obs, device)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
    
    env.close()
    print(f"Episode finished with reward: {total_reward}")
    print(f"Video saved in {VIDEO_DIR}")

if __name__ == "__main__":
    main()