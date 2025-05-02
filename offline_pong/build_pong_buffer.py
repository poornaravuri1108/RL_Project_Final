"""
Create an offline Pong dataset from a pre-trained SB3 DQN policy.
Saves d3rlpy MDPDataset -> pong_offline.h5
"""
import argparse, os
import numpy as np
import gymnasium as gym
import torch
from tqdm import tqdm

from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
from d3rlpy.dataset import MDPDataset
from huggingface_sb3 import load_from_hub

HF_REPO = "sb3/dqn-PongNoFrameskip-v4"
HF_FILE = "dqn-PongNoFrameskip-v4.zip"

def load_pretrained_dqn(device: str) -> DQN:
    path = load_from_hub(repo_id=HF_REPO, filename=HF_FILE)
    return DQN.load(path, device=device,
                    custom_objects={"optimize_memory_usage": False})

def main(args):
    torch.set_num_threads(os.cpu_count())
    model = load_pretrained_dqn(args.device)
    # Create Atari environment with SB3 wrappers for proper preprocessing
    env = gym.make("PongNoFrameskip-v4")                # NoFrameskip version for AtariWrapper
    env = AtariWrapper(env)                             # Apply Atari preprocessing (84x84 grayscale, frame skip, etc.):contentReference[oaicite:4]{index=4}
    env = DummyVecEnv([lambda: env])                    # Vectorize environment (required for frame stacking)
    env = VecFrameStack(env, n_stack=4)                 # Stack 4 frames (channels-last shape: 84x84x4)
    env = VecTransposeImage(env)                        # Transpose to channels-first (shape: 4x84x84):contentReference[oaicite:5]{index=5}

    obs = env.reset()  # SB3 VecEnv reset (returns initial observation, shape (1,4,84,84))
    obs_buf, act_buf, rew_buf, next_buf, term_buf = [], [], [], [], []
    pbar = tqdm(total=args.transitions, ncols=80)
    while len(obs_buf) < args.transitions:
        # Get action from the pre-trained model
        action, _state = model.predict(obs, deterministic=True)
        # Ensure action is in array form for VecEnv
        action = int(action) if isinstance(action, np.ndarray) else action
        obs_next, reward, done, infos = env.step([action])
        # VecEnv step returns batched outputs; extract the first (only) env
        obs_next = obs_next[0]
        reward   = float(reward[0])
        done     = bool(done[0])
        info     = infos[0]

        # If episode ended, use terminal observation for next state (from info), else use obs_next
        if done:
            # Get the final state (terminal observation) for logging
            final_obs = info.get("terminal_observation", obs_next)
            next_buf.append(final_obs) 
            term_buf.append(True)
        else:
            next_buf.append(obs_next)
            term_buf.append(False)
        # Log current transition
        obs_buf.append(obs[0])         # current state (remove batch dim)
        act_buf.append(action)
        rew_buf.append(reward)
        # Prepare next iteration
        obs = env.reset() if done else obs_next.reshape((1,)+obs_next.shape)
        # If we manually reset, env.reset() gives a batch of observations
        pbar.update(1)
    pbar.close()
    env.close()

    # Convert collected buffers to MDPDataset and save
    dataset = MDPDataset(
        observations      = np.array(obs_buf, dtype=np.float32),
        actions           = np.array(act_buf, dtype=np.int32),
        rewards           = np.array(rew_buf, dtype=np.float32),
        terminals         = np.array(term_buf, dtype=bool),
        next_observations = np.array(next_buf, dtype=np.float32)
    )
    dataset.dump(args.out)
    print(f"Saved {len(obs_buf):,} transitions ➜ {args.out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--transitions", type=int, default=1_000_000)
    parser.add_argument("--out", default="pong_offline.h5")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    main(parser.parse_args())



# """
# Create an offline Pong dataset from a pre‑trained SB3 DQN policy.
# Saves d3rlpy MDPDataset -> pong_offline.h5
# """
# import argparse, os, numpy as np, gymnasium as gym, torch
# from tqdm import tqdm
# from stable_baselines3 import DQN
# from d3rlpy.dataset import MDPDataset
# from huggingface_sb3 import load_from_hub           

# HF_REPO = "sb3/dqn-PongNoFrameskip-v4"
# HF_FILE = "dqn-PongNoFrameskip-v4.zip"

# def load_pretrained_dqn(device: str) -> DQN:
#     path = load_from_hub(repo_id=HF_REPO, filename=HF_FILE)
#     return DQN.load(path, device=device,
#                     custom_objects={"optimize_memory_usage": False})

# def main(args):
#     torch.set_num_threads(os.cpu_count())
#     model = load_pretrained_dqn(args.device)
#     env   = gym.make("ALE/Pong-v5", obs_type="grayscale", frameskip=4)
#     obs, _ = env.reset()

#     obs_buf, act_buf, rew_buf, next_buf, term_buf = [], [], [], [], []
#     pbar = tqdm(total=args.transitions, ncols=80)
#     while len(obs_buf) < args.transitions:
#         action, _state = model.predict(obs, deterministic=True)
#         nxt, reward, done, trunc, _ = env.step(action)

#         obs_buf.append(obs);         act_buf.append(action)
#         rew_buf.append(reward);      next_buf.append(nxt)
#         term_buf.append(done or trunc)

#         obs = nxt if not (done or trunc) else env.reset()[0]
#         pbar.update(1)
#     pbar.close(); env.close()

#     MDPDataset(
#         observations      = np.asarray(obs_buf,  np.float32),
#         actions           = np.asarray(act_buf,  np.int32),
#         rewards           = np.asarray(rew_buf,  np.float32),
#         terminals         = np.asarray(term_buf, bool),
#         next_observations = np.asarray(next_buf, np.float32)
#     ).dump(args.out)
#     print(f"Saved {len(obs_buf):,} transitions ➜ {args.out}")

# if __name__ == "__main__":
#     p = argparse.ArgumentParser()
#     p.add_argument("--transitions", type=int,   default=1_000_000)
#     p.add_argument("--out",         default="pong_offline.h5")
#     p.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
#     main(p.parse_args())
