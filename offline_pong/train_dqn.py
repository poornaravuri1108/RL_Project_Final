#!/usr/bin/env python3
"""
Robust offline‑DQN on Pong.
  – safe channel detection / conversion
  – gradient‑clipping
  – periodic evaluation
"""
import argparse, logging, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import h5py, gymnasium as gym
from tqdm import tqdm

# ---------------------------------------------------------------------------
class DQNNet(nn.Module):
    def __init__(self, c_in: int, n_actions: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c_in, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),   nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),   nn.ReLU(),
            nn.Flatten()
        )
        with torch.no_grad():
            dummy = torch.zeros(1, c_in, 84, 84)
            flat = self.conv(dummy).shape[1]
        self.value = nn.Sequential(nn.Linear(flat, 512), nn.ReLU(), nn.Linear(512, 1))
        self.adv   = nn.Sequential(nn.Linear(flat, 512), nn.ReLU(), nn.Linear(512, n_actions))

    def forward(self, x):
        x = x / 255.0
        z = self.conv(x)
        return self.value(z) + self.adv(z) - self.adv(z).mean(1, keepdim=True)

# ---------------------------------------------------------------------------
def to_chw(arr):
    """Return (C,84,84) numpy array regardless of original layout."""
    if arr.ndim == 2:                       # (H,W)
        return arr[np.newaxis, ...]         # → (1,H,W)
    if arr.shape[0] in {1,3,4}:             # already CHW
        return arr
    return np.transpose(arr, (2,0,1))       # HWC → CHW

# ---------------------------------------------------------------------------
def train(h5_path, steps, bs, device, seed):
    logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s",
                        level=logging.INFO)
    np.random.seed(seed); torch.manual_seed(seed)
    dev = torch.device(device)

    # --- load dataset -------------------------------------------------------
    h5 = h5py.File(h5_path, "r")
    obs_ds, act_ds, rew_ds, term_ds = (h5[k] for k in
                                       ("observations", "actions", "rewards", "terminals"))
    N = len(obs_ds)
    logging.info(f"Loaded {N} transitions")

    # infer channel‑count AFTER conversion
    in_ch = to_chw(obs_ds[0]).shape[0]
    logging.info(f"Inferred input channels = {in_ch}")

    # reward indices (for oversampling)
    rew_idx = np.where(rew_ds[:] != 0)[0]

    # --- networks & optim ---------------------------------------------------
    env = gym.make("ALE/Pong-v5", frameskip=1, obs_type="grayscale")
    env = gym.wrappers.AtariPreprocessing(env, frame_skip=4, grayscale_obs=True,
                                          screen_size=84, scale_obs=False)
    env = gym.wrappers.FrameStack(env, 4)
    n_actions = env.action_space.n

    net = DQNNet(in_ch, n_actions).to(dev)
    tgt = DQNNet(in_ch, n_actions).to(dev)
    tgt.load_state_dict(net.state_dict())
    optim = torch.optim.Adam(net.parameters(), lr=5e-5)
    gamma, sync_every = 0.99, 8_000

    # --- training loop ------------------------------------------------------
    log, t0 = [], time.time()
    for step in tqdm(range(1, steps+1), ncols=100, desc="DQN"):
        # sample mini‑batch (¾ uniform + ¼ reward transitions)
        n_sp = max(1, bs//4)
        idx  = np.concatenate([
            np.random.randint(0, N-1, bs-n_sp),
            np.random.choice(rew_idx[rew_idx < N-1], n_sp, replace=len(rew_idx)<n_sp)
        ])
        np.random.shuffle(idx)

        # build tensors ------------------------------------------------------
        obs_b, nxt_b, acts, rews, dones = [], [], [], [], []
        for i in idx:
            o, n = obs_ds[i], obs_ds[i+1]
            obs_b.append(to_chw(o)); nxt_b.append(to_chw(n))
            acts.append(int(act_ds[i]))
            rews.append(float(rew_ds[i])); dones.append(float(term_ds[i]))
        obs_t = torch.tensor(obs_b, dtype=torch.float32, device=dev)
        nxt_t = torch.tensor(nxt_b, dtype=torch.float32, device=dev)
        act_t = torch.tensor(acts,  dtype=torch.long,  device=dev)
        rew_t = torch.tensor(rews,  dtype=torch.float32, device=dev)
        don_t = torch.tensor(dones, dtype=torch.float32, device=dev)

        q_sa = net(obs_t).gather(1, act_t.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            na = net(nxt_t).argmax(1, keepdim=True)
            tgt_q = tgt(nxt_t).gather(1, na).squeeze(1)
            y = rew_t + gamma * tgt_q * (1-don_t)

        loss = F.smooth_l1_loss(q_sa, y)
        optim.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 10.0)
        optim.step()

        if step % sync_every == 0:
            tgt.load_state_dict(net.state_dict())
            logging.info(f"[{step}] target sync")

        if step % 20_000 == 0 or step == steps:
            # quick eval ------------------------------------------------------
            total = 0.0
            for _ in range(10):
                s, _ = env.reset(); ep = 0.0; done = False
                while not done:
                    s_t = torch.tensor(to_chw(s), dtype=torch.float32,
                                       device=dev).unsqueeze(0)
                    with torch.no_grad(): a = int(net(s_t).argmax(1))
                    s, r, done, _, _ = env.step(a); ep += r
                total += ep
            avg = total / 10
            log.append((step, avg, loss.item()))
            logging.info(f"[{step}] avg return {avg:.2f} | loss {loss.item():.4f}")

    torch.save(net.state_dict(), "dqn_pong.pt")
    with open("dqn_logs.csv", "w") as f:
        f.write("step,avg_return,loss\n")
        for s, r, l in log: f.write(f"{s},{r:.2f},{l:.4f}\n")
    h5.close(); logging.info("done")

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="pong_offline.h5")
    p.add_argument("--steps", type=int, default=500_000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    train(**vars(p.parse_args()))



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
