"""
Improved Offline PPO on Pong:
  - Behavior cloning warm-start (2 epochs)
  - Logit clamping & NaN safety for numerical stability
  - Gradient clipping (max_norm=0.5)
  - Reduced learning rate (1e-5)
"""
import argparse
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import h5py
import gymnasium as gym

# --- shared convolutional backbone (Nature CNN) ---
class CNNBackbone(nn.Sequential):
    def __init__(self, in_channels: int = 1):
        super().__init__(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),           nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),           nn.ReLU(),
            nn.Flatten()
        )
        # compute output dimension
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 84, 84)
            conv_out = self.forward(dummy)
        self.output_dim = conv_out.shape[1]

# --- ActorCritic network with two heads ---
class ActorCritic(nn.Module):
    def __init__(self, in_channels: int, n_actions: int):
        super().__init__()
        self.body = CNNBackbone(in_channels)
        self.policy_head = nn.Sequential(
            nn.Linear(self.body.output_dim, 512), nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        self.value_head = nn.Sequential(
            nn.Linear(self.body.output_dim, 512), nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x: torch.Tensor):
        # normalize pixel inputs
        x = x / 255.0
        z = self.body(x)
        logits = self.policy_head(z)             # (batch, n_actions)
        value  = self.value_head(z).squeeze(-1)  # (batch,)
        return logits, value

# --- compute discounted returns for an episode ---
def compute_returns(rewards: torch.Tensor, gamma: float = 0.99) -> torch.Tensor:
    returns = torch.zeros_like(rewards)
    G = 0.0
    for t in reversed(range(len(rewards))):
        G = rewards[t] + gamma * G
        returns[t] = G
    return returns

# --- simple behavior cloning warm-start on the offline buffer ---
def behavior_cloning(net, loader, optimizer, device, bc_epochs: int = 2):
    net.train()
    for _ in range(bc_epochs):
        for obs_batch, act_batch in loader:
            obs = obs_batch.to(device)
            acts = act_batch.to(device)
            logits, _ = net(obs)
            # clamp & nan-safe
            logits = logits.clamp(-10, 10)
            logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
            bc_loss = -torch.distributions.Categorical(logits=logits).log_prob(acts).mean()
            optimizer.zero_grad()
            bc_loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.5)
            optimizer.step()

# --- main training function ---
def train(dataset_path: str,
          epochs: int,
          device: str,
          clip: float = 0.2,
          lr: float = 1e-5,
          mb_size: int = 128,
          update_epochs: int = 2,
          seed: int = 42):
    torch.manual_seed(seed)
    device = torch.device(device)

    # build evaluation environment
    env = gym.make("ALE/Pong-v5", frameskip=1, obs_type="grayscale")
    env = gym.wrappers.AtariPreprocessing(
        env, frame_skip=4, grayscale_obs=True,
        screen_size=84, scale_obs=False
    )
    env = gym.wrappers.FrameStack(env, 4)
    n_actions = env.action_space.n

    # load entire offline dataset
    with h5py.File(dataset_path, "r") as h5:
        obs = np.array(h5["observations"])
        acts = np.array(h5["actions"])
        rews = np.array(h5["rewards"])
        terms= np.array(h5["terminals"])

    # preprocess observations → (N, C, 84, 84)
    if obs.ndim == 4 and obs.shape[-1] in [1, 4]:
        obs = obs.transpose(0, 3, 1, 2)
    elif obs.ndim == 3:
        obs = obs[:, None, :, :]
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
    act_t = torch.tensor(acts, dtype=torch.long, device=device)
    rew_t = torch.tensor(rews, dtype=torch.float32, device=device)

    # behavior cloning data loader
    bc_ds = TensorDataset(obs_t, act_t)
    bc_loader = DataLoader(bc_ds, batch_size=256, shuffle=True)

    # initialize network & optimizer
    net = ActorCritic(obs_t.shape[1], n_actions).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    # warm-start via behavior cloning
    behavior_cloning(net, bc_loader, optimizer, device, bc_epochs=2)

    # precompute old_logp & advantages over entire buffer
    with torch.no_grad():
        logits_all, vals_all = net(obs_t)
        logits_all = logits_all.clamp(-10, 10)
        logits_all = torch.nan_to_num(logits_all, nan=0.0, posinf=10.0, neginf=-10.0)
        dist_all = torch.distributions.Categorical(logits=logits_all)
        old_logp = dist_all.log_prob(act_t)

        returns_all = compute_returns(rew_t, gamma=0.99)
        advantages = returns_all - vals_all
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # offline PPO updates
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(len(obs_t), device=device)
        for _ in range(update_epochs):
            for i in range(0, len(obs_t), mb_size):
                idx = perm[i : i + mb_size]
                mb_obs = obs_t[idx]
                mb_act = act_t[idx]
                mb_old = old_logp[idx]
                mb_adv = advantages[idx]
                mb_ret = returns_all[idx]

                logits, vals = net(mb_obs)
                logits = logits.clamp(-10, 10)
                logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
                dist = torch.distributions.Categorical(logits=logits)
                new_logp = dist.log_prob(mb_act)

                ratio = (new_logp - mb_old).exp()
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss  = 0.5 * (mb_ret - vals).pow(2).mean()
                entropy_loss= -0.01 * dist.entropy().mean()
                loss = policy_loss + value_loss + entropy_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.5)
                optimizer.step()

        # evaluation after each epoch
        net.eval()
        total_reward = 0.0
        for _ in range(10):
            obs_e, _ = env.reset()
            done = False
            ep_reward = 0.0
            while not done:
                obs_tensor = torch.tensor(obs_e, dtype=torch.float32, device=device).unsqueeze(0) / 255.0
                if obs_tensor.ndim == 3:
                    obs_tensor = obs_tensor.unsqueeze(0)
                with torch.no_grad():
                    logits, _ = net(obs_tensor)
                    action = int(logits.argmax(dim=-1).item())
                obs_e, r, done, _, _ = env.step(action)
                ep_reward += r
            total_reward += ep_reward
        avg_return = total_reward / 10
        print(f"Epoch {epoch}: Avg Return = {avg_return:.2f}")
        net.train()

    # save final policy
    torch.save(net.state_dict(), "ppo_offline_improved.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="pong_offline.h5",
                        help="Path to the offline HDF5 dataset")
    parser.add_argument("--epochs",  type=int, default=10,
                        help="Number of PPO training epochs")
    parser.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to train on")
    parser.add_argument("--seed",    type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()
    train(args.dataset, args.epochs, args.device, seed=args.seed)



# #!/usr/bin/env python3
# """
# train_dqn.py

# Batched Offline DQN training on Pong with:
#   - 25% oversampling of ±1‐reward transitions
#   - Live logging of non‐zero reward fraction per batch
#   - Detailed tqdm postfix: loss, updates/sec, GPU memory, nz/bs
#   - Periodic evaluation and target‐net sync

# Usage:
#     python train_dqn.py \
#       --dataset pong_offline.h5 \
#       --steps 500000 \
#       --batch-size 64 \
#       --device cuda

# Outputs:
#     - dqn_pong.pt
#     - dqn_logs.csv       (step,eval/average_reward)
# """
# import argparse
# import logging
# import time

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import h5py
# import gymnasium as gym
# from tqdm import tqdm


# class DQNNetwork(nn.Module):
#     """Dueling DQN with Nature CNN backbone."""
#     def __init__(self, in_channels: int, n_actions: int):
#         super().__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels, 32, 8, stride=4), nn.ReLU(),
#             nn.Conv2d(32, 64, 4, stride=2),         nn.ReLU(),
#             nn.Conv2d(64, 64, 3, stride=1),         nn.ReLU(),
#             nn.Flatten()
#         )
#         with torch.no_grad():
#             dummy = torch.zeros(1, in_channels, 84, 84)
#             conv_out = self.conv(dummy).shape[1]
#         self.value_stream = nn.Sequential(
#             nn.Linear(conv_out, 512), nn.ReLU(), nn.Linear(512, 1)
#         )
#         self.adv_stream = nn.Sequential(
#             nn.Linear(conv_out, 512), nn.ReLU(), nn.Linear(512, n_actions)
#         )

#     def forward(self, x):
#         x = x / 255.0
#         feat = self.conv(x)
#         val  = self.value_stream(feat)
#         adv  = self.adv_stream(feat)
#         return val + adv - adv.mean(dim=1, keepdim=True)


# def train(dataset_path, steps, batch_size, device, seed):
#     # configure logging
#     logging.basicConfig(
#         format="%(asctime)s %(levelname)s: %(message)s",
#         level=logging.INFO
#     )
#     logging.info(f"Starting DQN training: dataset={dataset_path}, steps={steps}, "
#                  f"batch={batch_size}, device={device}")

#     # reproducibility
#     np.random.seed(seed)
#     torch.manual_seed(seed)

#     # build eval environment
#     eval_env = gym.make("ALE/Pong-v5", frameskip=1, obs_type="grayscale")
#     eval_env = gym.wrappers.AtariPreprocessing(
#         eval_env, frame_skip=4, grayscale_obs=True,
#         screen_size=84, scale_obs=False
#     )
#     eval_env = gym.wrappers.FrameStack(eval_env, 4)
#     n_actions = eval_env.action_space.n
#     logging.info(f"Eval env ready — n_actions={n_actions}")

#     # open HDF5 dataset
#     h5 = h5py.File(dataset_path, "r")
#     obs_ds  = h5["observations"]  # shape (N,4,84,84) or (N,84,84,4)
#     act_ds  = h5["actions"]
#     rew_ds  = h5["rewards"]
#     term_ds = h5["terminals"]
#     N = len(obs_ds)
#     logging.info(f"Loaded dataset with {N} transitions")

#     # infer input channels
#     sample = obs_ds[0]
#     if sample.ndim == 3 and sample.shape[0] in [1,4]:
#         in_ch = sample.shape[0]
#     elif sample.ndim == 3 and sample.shape[-1] in [1,4]:
#         in_ch = sample.shape[-1]
#     else:
#         in_ch = 1
#     logging.info(f"Inferred in_channels={in_ch}")

#     # precompute non-zero reward indices
#     all_rews = np.array(rew_ds[:])
#     reward_idxs = np.where(all_rews != 0)[0]
#     logging.info(f"Found {len(reward_idxs)} ±1 transitions for oversampling")

#     # build networks
#     dev = torch.device(device)
#     policy_net = DQNNetwork(in_ch, n_actions).to(dev)
#     target_net = DQNNetwork(in_ch, n_actions).to(dev)
#     target_net.load_state_dict(policy_net.state_dict())
#     target_net.eval()
#     optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-4)

#     gamma = 0.99
#     target_update_freq = 8000

#     # prepare logs
#     logs = [("step", "eval/average_reward")]
#     t_start = time.time()

#     # training loop
#     pbar = tqdm(range(1, steps+1), desc="DQN Training", ncols=100)
#     for step in pbar:
#         # sample 75% uniform, 25% from reward_idxs
#         num_special = max(1, batch_size // 4)
#         num_uniform = batch_size - num_special
#         idx_uniform = np.random.randint(0, N-1, size=num_uniform)
#         # ensure we don't pick the last index for idx+1
#         pool = reward_idxs[reward_idxs < N-1]
#         if len(pool) >= num_special:
#             idx_special = np.random.choice(pool, num_special, replace=False)
#         else:
#             idx_special = np.random.choice(pool, num_special, replace=True)
#         idxs = np.concatenate([idx_uniform, idx_special])
#         np.random.shuffle(idxs)

#         # load minibatch and permute channel-last if needed
#         obs_batch = np.stack([
#             np.transpose(obs_ds[i], (2,0,1)) if obs_ds[i].ndim==3 and obs_ds[i].shape[-1] in [1,4]
#             else obs_ds[i]
#             for i in idxs
#         ], axis=0)
#         nxt_batch = np.stack([
#             np.transpose(obs_ds[i+1], (2,0,1)) if obs_ds[i+1].ndim==3 and obs_ds[i+1].shape[-1] in [1,4]
#             else obs_ds[i+1]
#             for i in idxs
#         ], axis=0)
#         acts  = np.array([act_ds[i] for i in idxs], dtype=np.int64)
#         rews  = np.array([rew_ds[i] for i in idxs], dtype=np.float32)
#         dones = np.array([term_ds[i] for i in idxs], dtype=np.float32)

#         # to tensors
#         obs_t  = torch.tensor(obs_batch, dtype=torch.float32, device=dev)
#         nxt_t  = torch.tensor(nxt_batch, dtype=torch.float32, device=dev)
#         acts_t = torch.tensor(acts, dtype=torch.long, device=dev)
#         rews_t = torch.tensor(rews, dtype=torch.float32, device=dev)
#         done_t = torch.tensor(dones, dtype=torch.float32, device=dev)

#         # compute Q(s,a)
#         q_vals = policy_net(obs_t)
#         q_sa   = q_vals.gather(1, acts_t.unsqueeze(1)).squeeze(1)

#         # compute Double DQN target
#         with torch.no_grad():
#             q_next_policy = policy_net(nxt_t)
#             next_actions  = q_next_policy.argmax(dim=1, keepdim=True)
#             q_next_target = target_net(nxt_t).gather(1, next_actions).squeeze(1)
#             q_target      = rews_t + gamma * q_next_target * (1 - done_t)

#         # loss and backward
#         loss = F.smooth_l1_loss(q_sa, q_target)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         # sync target network
#         if step % target_update_freq == 0:
#             target_net.load_state_dict(policy_net.state_dict())
#             logging.info(f"[Step {step}] Target network synced")

#         # compute performance metrics
#         elapsed = time.time() - t_start
#         fps = step / elapsed
#         gpu_mem = ""
#         if dev.type == "cuda":
#             alloc = torch.cuda.memory_allocated(dev) / 1e9
#             resv  = torch.cuda.memory_reserved(dev) / 1e9
#             gpu_mem = f", GPU mem {alloc:.2f}G/{resv:.2f}G"
#         # non-zero reward fraction
#         nz_frac = (rews_t != 0).float().mean().item()
#         pbar.set_postfix({
#             "loss":  f"{loss.item():.3f}",
#             "fps":   f"{fps:.1f}{gpu_mem}",
#             "nz/bs": f"{int(nz_frac*batch_size)}/{batch_size}"
#         })

#         # periodic evaluation
#         if step % 100_000 == 0 or step == steps:
#             total_r = 0.0
#             trials = 5
#             for _ in range(trials):
#                 o, _ = eval_env.reset()
#                 done_e = False
#                 ep_r = 0.0
#                 while not done_e:
#                     o_np = np.array(o)
#                     if o_np.ndim==3 and o_np.shape[-1] in [1,4]:
#                         o_np = np.transpose(o_np, (2,0,1))
#                     o_t = torch.tensor(o_np, dtype=torch.float32, device=dev).unsqueeze(0)
#                     with torch.no_grad():
#                         qs = policy_net(o_t)
#                         a  = int(qs.argmax(dim=1).item())
#                     o, r, done_e, _, _ = eval_env.step(a)
#                     ep_r += r
#                 total_r += ep_r
#             avg_r = total_r / trials
#             logging.info(f"[Step {step}] Eval avg return = {avg_r:.2f}")
#             logs.append((step, avg_r))

#     # save model and logs
#     torch.save(policy_net.state_dict(), "dqn_pong.pt")
#     with open("dqn_logs.csv", "w") as f:
#         f.write("step,eval/average_reward\n")
#         for s, val in logs[1:]:
#             f.write(f"{s},{val:.2f}\n")

#     h5.close()
#     logging.info("Training complete. Model and logs saved.")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--dataset",    default="pong_offline.h5")
#     parser.add_argument("--steps",      type=int,   default=500_000)
#     parser.add_argument("--batch-size", type=int,   default=64)
#     parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
#     parser.add_argument("--seed",       type=int,   default=42)
#     args = parser.parse_args()
#     train(args.dataset, args.steps, args.batch_size, args.device, args.seed)
