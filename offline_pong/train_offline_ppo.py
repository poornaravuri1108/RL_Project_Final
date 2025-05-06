#!/usr/bin/env python3
"""
Memory‑safe offline‑PPO on Pong.
  – dataset lives on CPU
  – mini‑batches moved to GPU
  – behaviour‑cloning warm‑start
"""
import argparse, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import h5py, gymnasium as gym

# ───────────────── CNN + Actor‑Critic ───────────────────────────────────────
class CNNBackbone(nn.Sequential):
    def __init__(self, c_in):
        super().__init__(
            nn.Conv2d(c_in,32,8,4), nn.ReLU(),
            nn.Conv2d(32,64,4,2),   nn.ReLU(),
            nn.Conv2d(64,64,3,1),   nn.ReLU(),
            nn.Flatten()
        )
        with torch.no_grad():
            self.out_dim = self.forward(torch.zeros(1, c_in, 84, 84)).shape[1]

class ActorCritic(nn.Module):
    def __init__(self, c_in, n_act):
        super().__init__()
        self.body = CNNBackbone(c_in)
        self.pi = nn.Sequential(nn.Linear(self.body.out_dim,512), nn.ReLU(),
                                nn.Linear(512, n_act))
        self.v  = nn.Sequential(nn.Linear(self.body.out_dim,512), nn.ReLU(),
                                nn.Linear(512,1))

    def forward(self, x):
        z = self.body(x/255.0)
        return self.pi(z).clamp(-10,10), self.v(z).squeeze(-1)

# ───────────────── helpers ──────────────────────────────────────────────────
def to_chw(arr):
    return arr[np.newaxis] if arr.ndim==2 else (
        arr if arr.shape[0] in {1,3,4} else np.transpose(arr,(2,0,1)))

def compute_returns(rew, gamma=0.99):
    out = np.zeros_like(rew, dtype=np.float32); G = 0.0
    for t in reversed(range(len(rew))):
        G = rew[t] + gamma*G; out[t] = G
    return out

# ───────────────── training ────────────────────────────────────────────────
def train(h5_path, epochs, device, seed=42, lr=1e-5,
          mb=128, clip=0.2, upd_epochs=2):
    torch.manual_seed(seed); np.random.seed(seed)
    dev = torch.device(device)

    # -------- dataset on CPU ------------------------------------------------
    with h5py.File(h5_path,"r") as h5:
        obs = np.array(h5["observations"]); acts = np.array(h5["actions"])
        rews= np.array(h5["rewards"]);      terms= np.array(h5["terminals"])

    obs = np.stack([to_chw(o) for o in obs])           # (N,C,84,84)
    ret = compute_returns(rews)                         # (N,)

    c_in = obs.shape[1]
    ds = TensorDataset(torch.from_numpy(obs).float(),   # still on CPU
                       torch.from_numpy(acts).long(),
                       torch.from_numpy(ret).float())
    loader = DataLoader(ds, batch_size=mb, shuffle=True, drop_last=True)

    # -------- env for eval --------------------------------------------------
    env = gym.make("ALE/Pong-v5", frameskip=1, obs_type="grayscale")
    env = gym.wrappers.AtariPreprocessing(env, frame_skip=4, grayscale_obs=True,
                                          screen_size=84, scale_obs=False)
    env = gym.wrappers.FrameStack(env,4); n_act = env.action_space.n

    # -------- network & opt -------------------------------------------------
    net = ActorCritic(c_in, n_act).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    # -------- behaviour‑cloning warm‑start ----------------------------------
    for _ in range(2):          # 2 epochs BC
        for o,a,_ in loader:
            o,a = o.to(dev), a.to(dev)
            logit,_ = net(o); dist = torch.distributions.Categorical(logits=logit)
            loss = -dist.log_prob(a).mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5); opt.step()

    # -------- PPO passes ----------------------------------------------------
    for ep in range(1, epochs+1):
        # pre‑compute old log‑probs & advantages (streamed)
        old_logp, adv_buf = [], []
        with torch.no_grad():
            for o, a, R in loader:
                o_dev = o.to(dev)
                logit, val = net(o_dev)
                dist = torch.distributions.Categorical(logits=logit)
                old_logp.append(dist.log_prob(a.to(dev)).cpu())
                adv_buf.append((R - val.cpu()).numpy())
        old_logp = torch.cat(old_logp)
        adv = torch.tensor(np.concatenate(adv_buf))
        adv = (adv - adv.mean())/(adv.std()+1e-8)

        # PPO updates
        for _ in range(upd_epochs):
            for o,a,R,ol,ad in loader:
                idx = torch.randperm(len(o))
                o,a,R = o[idx], a[idx], R[idx]
                ol,ad = old_logp[idx], adv[idx]

                o,a,R,ol,ad = o.to(dev), a.to(dev), R.to(dev), ol.to(dev), ad.to(dev)
                logit,val = net(o)
                dist = torch.distributions.Categorical(logits=logit)
                ratio = torch.exp(dist.log_prob(a) - ol)
                s1 = ratio*ad; s2 = torch.clamp(ratio,1-clip,1+clip)*ad
                pol_loss = -torch.min(s1,s2).mean()
                val_loss = 0.5*(R - val).pow(2).mean()
                ent_loss = -0.01*dist.entropy().mean()
                loss = pol_loss + val_loss + ent_loss
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(),0.5); opt.step()

        # ---------- quick eval ----------------------------------------------
        net.eval(); tot=0.0
        for _ in range(10):
            s,_ = env.reset(); ep_r=0.0; done=False
            while not done:
                st = torch.tensor(to_chw(s),dtype=torch.float32,
                                  device=dev).unsqueeze(0)
                with torch.no_grad(): act = int(net(st)[0].argmax(1))
                s,r,done,_,_ = env.step(act); ep_r+=r
            tot+=ep_r
        print(f"Epoch {ep}: avg return {tot/10:.2f}")
        net.train()

    torch.save(net.state_dict(),"ppo_offline.pt")

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--dataset", default="pong_offline.h5")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    train(**vars(p.parse_args()))




# import argparse
# import collections
# import numpy as np
# import torch
# from torch import nn
# from torch.utils.data import DataLoader
# import h5py
# import gymnasium as gym

# # Convolutional backbone (Nature CNN) for 84x84 inputs
# class CNNBackbone(nn.Sequential):
#     def __init__(self, in_channels: int = 1):
#         super().__init__(
#             nn.Conv2d(in_channels, 32, kernel_size=8, stride=4), nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
#             nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
#             nn.Flatten()
#         )
#         # Calculate output dimension for a single 84x84 input
#         with torch.no_grad():
#             dummy = torch.zeros(1, in_channels, 84, 84)
#             conv_out = self.forward(dummy)
#         self.output_dim = conv_out.shape[1]

# class ActorCritic(nn.Module):
#     def __init__(self, in_channels: int, n_actions: int):
#         super().__init__()
#         self.body = CNNBackbone(in_channels)
#         self.policy_head = nn.Sequential(
#             nn.Linear(self.body.output_dim, 512), nn.ReLU(),
#             nn.Linear(512, n_actions)
#         )
#         self.value_head = nn.Sequential(
#             nn.Linear(self.body.output_dim, 512), nn.ReLU(),
#             nn.Linear(512, 1)
#         )

#     def forward(self, x: torch.Tensor):
#         # Normalize pixel inputs
#         x = x / 255.0
#         z = self.body(x)
#         logits = self.policy_head(z)       # unnormalized action logits
#         value = self.value_head(z).squeeze(-1)  # state value
#         return logits, value

# def compute_returns(rewards: torch.Tensor, gamma: float = 0.99) -> torch.Tensor:
#     """Compute discounted returns for a sequence of rewards (1D tensor)."""
#     returns = torch.zeros_like(rewards)
#     G = 0.0
#     # Compute returns backwards
#     for t in reversed(range(len(rewards))):
#         G = rewards[t] + gamma * G
#         returns[t] = G
#     return returns

# def train(dataset_path: str, epochs: int, device: str,
#           clip: float = 0.2, lr: float = 2.5e-4,
#           minibatch_size: int = 256, update_epochs: int = 4, seed: int = 42):
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     device = torch.device(device)

#     # Initialize environment for evaluation
#     env = gym.make("ALE/Pong-v5", frameskip=1, obs_type="grayscale", render_mode=None)
#     env = gym.wrappers.AtariPreprocessing(env, frame_skip=4, grayscale_obs=True,
#                                           grayscale_newaxis=False, screen_size=84, scale_obs=False)
#     env = gym.wrappers.FrameStack(env, 4)
#     n_actions = env.action_space.n

#     # Determine observation channels from dataset
#     with h5py.File(dataset_path, 'r') as h5:
#         sample = h5['observations'][0]
#     sample = np.array(sample)
#     if sample.ndim == 3:
#         if sample.shape[0] in [1, 4]:
#             in_channels = sample.shape[0]
#         elif sample.shape[-1] in [1, 4]:
#             in_channels = sample.shape[-1]
#         else:
#             in_channels = 1
#     else:
#         in_channels = 1

#     net = ActorCritic(in_channels, n_actions).to(device)
#     optimizer = torch.optim.Adam(net.parameters(), lr=lr)
#     scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

#     # For logging results
#     log_data = [("step", "average_reward")]
#     total_updates = 0

#     # Training loop over epochs and episodes
#     for epoch in range(1, epochs + 1):
#         # Open dataset file for reading episodes sequentially
#         with h5py.File(dataset_path, 'r') as h5:
#             obs_ds = h5['observations']
#             act_ds = h5['actions']
#             rew_ds = h5['rewards']
#             term_ds = h5['terminals']
#             data_len = len(obs_ds)
#             idx = 0
#             # Iterate through each episode in the dataset
#             while idx < data_len:
#                 # Collect one episode
#                 obs_list, act_list, rew_list = [], [], []
#                 done = False
#                 while idx < data_len and not done:
#                     obs_list.append(np.array(obs_ds[idx]))
#                     act_list.append(act_ds[idx])
#                     rew_list.append(rew_ds[idx])
#                     done = bool(term_ds[idx])
#                     idx += 1
#                 # Convert episode to tensors
#                 obs_ep = torch.tensor(np.array(obs_list), dtype=torch.float32, device=device)
#                 act_ep = torch.tensor(act_list, dtype=torch.long, device=device)
#                 rew_ep = torch.tensor(rew_list, dtype=torch.float32, device=device)
#                 # Add channel dim to observations if needed
#                 if obs_ep.ndim == 3:
#                     # If obs_ep shape is (T, H, W, C) or (T, C, H, W)
#                     if obs_ep.shape[1] not in [1, 4] and obs_ep.shape[-1] in [1, 4]:
#                         # Permute from (T, H, W, C) to (T, C, H, W)
#                         obs_ep = obs_ep.permute(0, 3, 1, 2)
#                     elif obs_ep.shape[1] not in [1, 4] and obs_ep.shape[-1] not in [1, 4]:
#                         obs_ep = obs_ep.unsqueeze(1)  # (T, H, W) -> (T,1,H,W)
#                 elif obs_ep.ndim == 4 and obs_ep.shape[1] not in [1, 4] and obs_ep.shape[-1] in [1, 4]:
#                     obs_ep = obs_ep.permute(0, 3, 1, 2)
#                 # Compute old policy log-probs and state values (no grad)
#                 with torch.no_grad():
#                     logits, vals = net(obs_ep)
#                     dist = torch.distributions.Categorical(logits=logits)
#                     old_log_prob = dist.log_prob(act_ep)
#                     # Advantage estimation: compute returns and subtract values
#                     returns = compute_returns(rew_ep, gamma=0.99).to(device)
#                     advantage = returns - vals
#                     advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
#                 # PPO updates for this episode
#                 for _ in range(update_epochs):
#                     # Shuffle indices for mini-batches
#                     perm = torch.randperm(len(obs_ep), device=device)
#                     for start in range(0, len(obs_ep), minibatch_size):
#                         end = start + minibatch_size
#                         idx_mb = perm[start:end]
#                         mb_obs = obs_ep[idx_mb]
#                         mb_act = act_ep[idx_mb]
#                         mb_adv = advantage[idx_mb]
#                         mb_ret = returns[idx_mb]
#                         mb_old_logp = old_log_prob[idx_mb]

#                         # Forward pass with AMP (if enabled)
#                         with torch.cuda.amp.autocast(enabled=(scaler is not None)):
#                             logits, values = net(mb_obs)
#                             dist = torch.distributions.Categorical(logits=logits)
#                             new_logp = dist.log_prob(mb_act)
#                             # Calculate PPO loss components
#                             log_ratio = new_logp - mb_old_logp
#                             ratio = torch.exp(log_ratio)
#                             surr1 = ratio * mb_adv
#                             surr2 = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * mb_adv
#                             policy_loss = -torch.min(surr1, surr2).mean()
#                             value_loss = 0.5 * (mb_ret - values).pow(2).mean()
#                             entropy_loss = -0.01 * dist.entropy().mean()
#                             loss = policy_loss + value_loss + entropy_loss
#                         # Backpropagate loss
#                         optimizer.zero_grad()
#                         if scaler:
#                             scaler.scale(loss).backward()
#                             scaler.step(optimizer)
#                             scaler.update()
#                         else:
#                             loss.backward()
#                             optimizer.step()
#                         total_updates += 1
#         # End of epoch: evaluate policy on environment
#         total_reward = 0.0
#         eval_episodes = 10
#         net.eval()
#         for _ in range(eval_episodes):
#             obs, _ = env.reset()
#             obs = np.array(obs)
#             done_flag = False
#             episode_reward = 0.0
#             while not done_flag:
#                 obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
#                 if obs_tensor.ndim == 3 and obs_tensor.shape[0] not in [1, 4]:
#                     obs_tensor = obs_tensor.permute(2, 0, 1)
#                 elif obs_tensor.ndim == 2:
#                     obs_tensor = obs_tensor.unsqueeze(0)
#                 with torch.no_grad():
#                     logits, _ = net(obs_tensor.unsqueeze(0))
#                     action = int(torch.argmax(logits, dim=-1).item())
#                 obs, reward, done_flag, _, _ = env.step(action)
#                 obs = np.array(obs)
#                 episode_reward += reward
#             total_reward += episode_reward
#         net.train()
#         avg_return = total_reward / eval_episodes
#         print(f"Epoch {epoch}: PPO average return = {avg_return:.2f}")
#         log_data.append((total_updates, avg_return))
#     # Save model after training
#     torch.save(net.state_dict(), "ppo_offline.pt")
#     # Write training log to CSV
#     with open("ppo_logs.csv", "w") as f:
#         f.write("step,average_reward\n")
#         for step, rew in log_data[1:]:
#             f.write(f"{step},{rew:.2f}\n")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--dataset", default="pong_offline.h5", help="Path to HDF5 offline dataset")
#     parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs for PPO")
#     parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to train on")
#     parser.add_argument("--seed", type=int, default=42, help="Random seed")
#     args = parser.parse_args()
#     train(args.dataset, args.epochs, args.device, seed=args.seed)
