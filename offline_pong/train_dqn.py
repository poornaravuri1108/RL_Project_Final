#!/usr/bin/env python3
"""
train_dqn.py

Batched Offline DQN training on Pong from an HDF5 dataset.
Shows detailed GPU/CPU usage, loss, and throughput in the console.

Usage:
    python train_dqn.py \
      --dataset pong_offline.h5 \
      --steps 500000 \
      --batch-size 64 \
      --device cuda

Outputs:
    - dqn_pong.pt
    - dqn_logs.csv       (step,eval/average_reward)
"""
import argparse
import logging
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
import gymnasium as gym
from tqdm import tqdm

class DQNNetwork(nn.Module):
    """Dueling DQN with Nature CNN backbone."""
    def __init__(self, in_channels: int, n_actions: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),         nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),         nn.ReLU(),
            nn.Flatten()
        )
        # find conv output dim
        with torch.no_grad():
            d = self.conv(torch.zeros(1, in_channels, 84, 84)).shape[1]
        self.value_stream = nn.Sequential(
            nn.Linear(d, 512), nn.ReLU(), nn.Linear(512, 1)
        )
        self.adv_stream = nn.Sequential(
            nn.Linear(d, 512), nn.ReLU(), nn.Linear(512, n_actions)
        )

    def forward(self, x):
        x = x / 255.0
        feat = self.conv(x)
        val  = self.value_stream(feat)
        adv  = self.adv_stream(feat)
        return val + adv - adv.mean(dim=1, keepdim=True)


def train(dataset_path, steps, batch_size, device, seed):
    # set up logging
    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(message)s",
        level=logging.INFO
    )
    logging.info(f"Training DQN — dataset: {dataset_path}, steps: {steps}, batch: {batch_size}, device: {device}")

    # reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    # build evaluation env
    eval_env = gym.make("ALE/Pong-v5", frameskip=1, obs_type="grayscale")
    eval_env = gym.wrappers.AtariPreprocessing(
        eval_env, frame_skip=4, grayscale_obs=True, screen_size=84, scale_obs=False
    )
    eval_env = gym.wrappers.FrameStack(eval_env, 4)
    n_actions = eval_env.action_space.n
    logging.info(f"Eval env ready — n_actions={n_actions}")

    # load HDF5 dataset
    h5 = h5py.File(dataset_path, "r")
    obs_ds    = h5["observations"]  # shape (N,4,84,84) or (N,84,84,4)
    act_ds    = h5["actions"]
    rew_ds    = h5["rewards"]
    term_ds   = h5["terminals"]
    N = len(obs_ds)
    logging.info(f"Loaded dataset with {N} transitions")

    # detect channel layout
    sample = obs_ds[0]
    if sample.ndim == 3 and sample.shape[0] in [1,4]:
        in_ch = sample.shape[0]
    elif sample.ndim == 3 and sample.shape[-1] in [1,4]:
        in_ch = sample.shape[-1]
    else:
        in_ch = 1
    logging.info(f"Inferred in_channels={in_ch}")

    # build networks
    dev = torch.device(device)
    policy_net = DQNNetwork(in_ch, n_actions).to(dev)
    target_net = DQNNetwork(in_ch, n_actions).to(dev)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-4)
    gamma = 0.99
    target_update_freq = 8000

    # training metrics
    logs = [("step", "eval/average_reward")]
    t0 = time.time()

    # main loop with tqdm
    pbar = tqdm(range(1, steps+1), desc="DQN Training", ncols=100)
    for step in pbar:
        # sample a minibatch of indices (avoiding last index)
        idx = np.random.randint(0, N-1, size=batch_size)
        obs_np = obs_ds[idx]           # (B,4,84,84) or (B,84,84,4)
        nxt_np = obs_ds[idx+1]
        acts   = act_ds[idx]
        rews   = rew_ds[idx]
        dones  = term_ds[idx]

        # permute channel-last -> channel-first if needed
        if obs_np.ndim == 4 and obs_np.shape[-1] in [1,4]:
            obs_np = np.transpose(obs_np, (0,3,1,2))
            nxt_np = np.transpose(nxt_np, (0,3,1,2))

        # convert to tensors
        obs_t = torch.tensor(obs_np, dtype=torch.float32, device=dev)
        nxt_t = torch.tensor(nxt_np, dtype=torch.float32, device=dev)
        acts_t= torch.tensor(acts,   dtype=torch.long,    device=dev)
        rews_t= torch.tensor(rews,   dtype=torch.float32, device=dev)
        dones_t= torch.tensor(dones, dtype=torch.float32, device=dev)

        # Q(s,a)
        q_vals = policy_net(obs_t)                     # (B,A)
        q_sa   = q_vals.gather(1, acts_t.unsqueeze(1)).squeeze(1)

        # Double DQN target: select with policy_net, evaluate with target_net
        with torch.no_grad():
            q_pol_next = policy_net(nxt_t)
            a_pol_next = q_pol_next.argmax(dim=1, keepdim=True)
            q_tgt_next = target_net(nxt_t).gather(1, a_pol_next).squeeze(1)
            target = rews_t + gamma * q_tgt_next * (1.0 - dones_t)

        # loss and update
        loss = F.smooth_l1_loss(q_sa, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # target network sync
        if step % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())
            logging.info(f"Step {step}: target network synced")

        # measure performance & GPU mem
        elapsed = time.time() - t0
        fps = step / elapsed
        if dev.type == 'cuda':
            mem_alloc = torch.cuda.memory_allocated(dev) / 1e9
            mem_reserved = torch.cuda.memory_reserved(dev) / 1e9
        else:
            mem_alloc = mem_reserved = 0.0

        pbar.set_postfix({
            "loss": f"{loss.item():.3f}",
            "fps":  f"{fps:.1f}",
            "gpu_mem": f"{mem_alloc:.2f}G/{mem_reserved:.2f}G"
        })

        # evaluation every 100k steps
        if step % 100_000 == 0 or step == steps:
            total = 0.0
            trials = 5
            for _ in range(trials):
                o, _ = eval_env.reset()
                done_e = False
                score = 0.0
                while not done_e:
                    o_np = np.array(o)
                    if o_np.ndim==3 and o_np.shape[-1] in [1,4]:
                        o_np = np.transpose(o_np,(2,0,1))
                    o_tsn = torch.tensor(o_np, dtype=torch.float32, device=dev).unsqueeze(0)
                    with torch.no_grad():
                        qs = policy_net(o_tsn)
                        a  = int(qs.argmax(dim=1).item())
                    o, r, done_e, _, _ = eval_env.step(a)
                    score += r
                total += score
            avg = total / trials
            logging.info(f"Step {step}: eval avg return = {avg:.2f}")
            logs.append((step, avg))

    # save model + logs
    torch.save(policy_net.state_dict(), "dqn_pong.pt")
    with open("dqn_logs.csv","w") as f:
        f.write("step,eval/average_reward\n")
        for s,v in logs[1:]:
            f.write(f"{s},{v:.2f}\n")

    h5.close()
    logging.info("Training complete. Model and logs saved.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",    default="pong_offline.h5")
    p.add_argument("--steps",      type=int,   default=500_000)
    p.add_argument("--batch-size", type=int,   default=64)
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed",       type=int,   default=42)
    args = p.parse_args()
    train(args.dataset, args.steps, args.batch_size, args.device, args.seed)
