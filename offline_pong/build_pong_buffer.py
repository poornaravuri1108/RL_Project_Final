"""
build_pong_buffer.py

Generate an offline Pong dataset by rolling out a pre-trained SB3 DQN policy.
Outputs an HDF5 file with datasets: observations (uint8 [0–255]), actions, rewards, terminals.

Usage:
    python build_pong_buffer.py --transitions 500000 --out pong_offline.h5 --device cuda

The script automatically uses GPU if available (via --device cuda), otherwise CPU.
"""
import argparse
import os
import warnings

import numpy as np
import torch
import h5py
from tqdm import tqdm

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from huggingface_sb3 import load_from_hub

# Hugging Face SB3 model repo for pretrained DQN
HF_REPO = "sb3/dqn-PongNoFrameskip-v4"
HF_FILE = "dqn-PongNoFrameskip-v4.zip"

def load_pretrained_dqn(device: str) -> DQN:
    """
    Download (if needed) and load the pre-trained DQN policy.
    Silences harmless SB3 loading warnings.
    """
    warnings.filterwarnings(
        "ignore",
        message=".*loaded a model that was trained using OpenAI Gym.*",
        category=UserWarning
    )
    path = load_from_hub(repo_id=HF_REPO, filename=HF_FILE)
    return DQN.load(path, device=device, custom_objects={"optimize_memory_usage": False})

def main(transitions: int, out_path: str, device: str):
    # Load pretrained DQN
    model = load_pretrained_dqn(device)
    torch.set_num_threads(os.cpu_count() or 1)

    # Observation shape: (1, 4, 84, 84)
    env = make_atari_env("PongNoFrameskip-v4", n_envs=1, seed=0)
    env = VecFrameStack(env, n_stack=4)

    # Prepare HDF5 file for streaming writes
    with h5py.File(out_path, "w") as f:
        obs = env.reset()  # shape (1, 4, 84, 84), dtype uint8
        obs_dtype = obs.dtype
        n_envs, C, H, W = obs.shape

        # Create datasets
        obs_dset       = f.create_dataset("observations", shape=(transitions, C, H, W), dtype=obs_dtype)
        actions_dset   = f.create_dataset("actions",      shape=(transitions,),        dtype=np.int8)
        rewards_dset   = f.create_dataset("rewards",      shape=(transitions,),        dtype=np.float32)
        terminals_dset = f.create_dataset("terminals",    shape=(transitions,),        dtype=bool)
        f.create_dataset("discrete_action", data=True)

        pbar = tqdm(total=transitions, desc="Generating dataset", ncols=80)
        count = 0

        while count < transitions:
            if np.random.random() < 0.3:
                a = int(np.random.randint(0, env.action_space.n))
            else:
                deterministic = np.random.random() < 0.7 
                action, _ = model.predict(obs, deterministic=deterministic)
                a = int(action)  

            obs_next, reward, done, infos = env.step([a])

            obs_dset[count]       = obs[0]        
            actions_dset[count]   = a
            rewards_dset[count]   = float(reward[0])
            terminals_dset[count] = bool(done[0])

            count += 1
            pbar.update(1)

            if done[0]:
                obs = env.reset()
            else:
                obs = obs_next

        pbar.close()
    env.close()

    print(f"\nDone — saved {count:,} transitions to '{out_path}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transitions", "-n", type=int, default=500_000,
        help="Number of transitions to collect"
    )
    parser.add_argument(
        "--out", "-o", default="pong_offline.h5",
        help="Output HDF5 file path"
    )
    parser.add_argument(
        "--device", "-d", default="cuda" if torch.cuda.is_available() else "cpu",
        help="PyTorch device (cuda or cpu)"
    )
    args = parser.parse_args()
    main(args.transitions, args.out, args.device)