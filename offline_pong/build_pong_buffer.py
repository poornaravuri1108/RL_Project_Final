"""
Create an offline Pong dataset from a pre‑trained SB3 DQN policy.
Saves d3rlpy MDPDataset -> pong_offline.h5
"""
import argparse, os, numpy as np, gymnasium as gym, torch
from tqdm import tqdm
from stable_baselines3 import DQN
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
    env   = gym.make("ALE/Pong-v5", obs_type="grayscale", frameskip=4)
    obs, _ = env.reset()

    obs_buf, act_buf, rew_buf, next_buf, term_buf = [], [], [], [], []
    pbar = tqdm(total=args.transitions, ncols=80)
    while len(obs_buf) < args.transitions:
        action, _state = model.predict(obs, deterministic=True)
        nxt, reward, done, trunc, _ = env.step(action)

        obs_buf.append(obs);         act_buf.append(action)
        rew_buf.append(reward);      next_buf.append(nxt)
        term_buf.append(done or trunc)

        obs = nxt if not (done or trunc) else env.reset()[0]
        pbar.update(1)
    pbar.close(); env.close()

    MDPDataset(
        observations      = np.asarray(obs_buf,  np.float32),
        actions           = np.asarray(act_buf,  np.int32),
        rewards           = np.asarray(rew_buf,  np.float32),
        terminals         = np.asarray(term_buf, bool),
        next_observations = np.asarray(next_buf, np.float32)
    ).dump(args.out)
    print(f"Saved {len(obs_buf):,} transitions ➜ {args.out}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--transitions", type=int,   default=1_000_000)
    p.add_argument("--out",         default="pong_offline.h5")
    p.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    main(p.parse_args())
