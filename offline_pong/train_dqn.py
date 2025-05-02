"""
train_dqn.py

Offline DQN baseline training on a Pong dataset stored as HDF5:
  - observations: uint8 [0–255], shape=(N,4,84,84) **or** (N,84,84,4)
  - actions: int8,      shape=(N,)
  - rewards: float32,   shape=(N,)
  - terminals: bool,    shape=(N,)

Usage:
    python train_dqn.py --dataset pong_offline.h5 --steps 1000000 --device cuda

Saves:
  - dqn_pong.pt
  - dqn_logs.csv (step,eval/average_reward)
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
import gymnasium as gym


class DQNNetwork(nn.Module):
    def __init__(self, in_channels: int, n_actions: int):
        super().__init__()
        # Nature CNN
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),         nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),         nn.ReLU(),
            nn.Flatten()
        )
        # compute conv output dim
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 84, 84)
            conv_out = self.conv(dummy).shape[1]
        # dueling heads
        self.value_stream = nn.Sequential(
            nn.Linear(conv_out, 512), nn.ReLU(),
            nn.Linear(512, 1)
        )
        self.adv_stream = nn.Sequential(
            nn.Linear(conv_out, 512), nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, x):
        # normalize
        x = x / 255.0
        feat = self.conv(x)
        val = self.value_stream(feat)           # (B,1)
        adv = self.adv_stream(feat)             # (B,A)
        # combine
        q = val + adv - adv.mean(dim=1, keepdim=True)
        return q


def train(dataset_path, steps, device, seed):
    # reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    # build eval env
    eval_env = gym.make("ALE/Pong-v5", frameskip=1, obs_type="grayscale")
    eval_env = gym.wrappers.AtariPreprocessing(
        eval_env, frame_skip=4, grayscale_obs=True,
        screen_size=84, scale_obs=False
    )
    eval_env = gym.wrappers.FrameStack(eval_env, 4)
    n_actions = eval_env.action_space.n

    # open HDF5
    h5 = h5py.File(dataset_path, "r")
    obs_ds    = h5["observations"]
    act_ds    = h5["actions"]
    rew_ds    = h5["rewards"]
    term_ds   = h5["terminals"]
    dataset_size = len(obs_ds)

    # detect channels from first sample
    sample = obs_ds[0]  # could be shape (4,84,84) or (84,84,4)
    if sample.ndim == 3 and sample.shape[0] in [1,4]:
        in_channels = sample.shape[0]
    elif sample.ndim == 3 and sample.shape[-1] in [1,4]:
        in_channels = sample.shape[-1]
    else:
        in_channels = 1

    # model & target
    device = torch.device(device)
    policy_net = DQNNetwork(in_channels, n_actions).to(device)
    target_net = DQNNetwork(in_channels, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-4)
    gamma = 0.99
    target_update = 8000

    # logging
    logs = [("step","eval/average_reward")]

    for step in range(1, steps+1):
        # sample transition idx (ensure idx+1 < dataset_size)
        idx = np.random.randint(0, dataset_size-1)
        obs_np = obs_ds[idx]
        act    = int(act_ds[idx])
        rew    = float(rew_ds[idx])
        done   = bool(term_ds[idx])

        # get next-state Q
        if done:
            q_next = 0.0
        else:
            nxt_np = obs_ds[idx+1]
            # permute if channel-last
            if nxt_np.ndim==3 and nxt_np.shape[-1] in [1,4]:
                nxt_np = np.transpose(nxt_np, (2,0,1))
            nxt_t = torch.tensor(nxt_np, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                q_pol = policy_net(nxt_t)
                a_pol = int(q_pol.argmax(dim=1).item())
                q_tgt = target_net(nxt_t)
                q_next = float(q_tgt[0,a_pol].item())

        # prepare current state tensor
        # permute if channel-last
        if obs_np.ndim==3 and obs_np.shape[-1] in [1,4]:
            obs_np = np.transpose(obs_np, (2,0,1))
        obs_t = torch.tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)

        # compute Q(s,a)
        q_vals = policy_net(obs_t)
        q_sa   = q_vals[0, act]
        target = rew + gamma * q_next

        # loss & optimize
        loss = F.smooth_l1_loss(q_sa, torch.tensor(target, device=device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update target
        if step % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # evaluate periodically
        if step % 100_000 == 0 or step == steps:
            total_score = 0.0
            trials = 10
            for _ in range(trials):
                o, _ = eval_env.reset()
                done_e = False
                score = 0.0
                while not done_e:
                    o_np = np.array(o)
                    if o_np.ndim==3 and o_np.shape[-1] in [1,4]:
                        o_np = np.transpose(o_np,(2,0,1))
                    o_t = torch.tensor(o_np, dtype=torch.float32, device=device).unsqueeze(0)
                    with torch.no_grad():
                        qs = policy_net(o_t)
                        a  = int(qs.argmax(dim=1).item())
                    o, r, done_e, _, _ = eval_env.step(a)
                    score += r
                total_score += score
            ave = total_score / trials
            print(f"Step {step}: DQN avg return = {ave:.2f}")
            logs.append((step, ave))

    # save and log
    torch.save(policy_net.state_dict(), "dqn_pong.pt")
    with open("dqn_logs.csv","w") as f:
        f.write("step,eval/average_reward\n")
        for s, val in logs[1:]:
            f.write(f"{s},{val:.2f}\n")

    h5.close()


if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="pong_offline.h5")
    p.add_argument("--steps",   type=int,   default=1_000_000)
    p.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed",    type=int,   default=42)
    args = p.parse_args()
    train(args.dataset, args.steps, args.device, args.seed)
