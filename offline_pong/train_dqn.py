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
    def __init__(self, in_channels: int, n_actions: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),         nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),         nn.ReLU(),
            nn.Flatten()
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 84, 84)
            conv_out = self.conv(dummy).shape[1]
        self.value_stream = nn.Sequential(
            nn.Linear(conv_out, 512), nn.ReLU(), nn.Linear(512, 1)
        )
        self.adv_stream = nn.Sequential(
            nn.Linear(conv_out, 512), nn.ReLU(), nn.Linear(512, n_actions)
        )

    def forward(self, x):
        x = x / 255.0
        feat = self.conv(x)
        val  = self.value_stream(feat)
        adv  = self.adv_stream(feat)
        return val + adv - adv.mean(dim=1, keepdim=True)


def train(dataset_path, steps, batch_size, device, seed, cql_alpha=1.0, use_cql=True, td_clip=10.0):
    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(message)s",
        level=logging.INFO
    )
    logging.info(f"Starting DQN training: dataset={dataset_path}, steps={steps}, "
                 f"batch={batch_size}, device={device}")

    np.random.seed(seed)
    torch.manual_seed(seed)

    eval_env = gym.make("ALE/Pong-v5", frameskip=1, obs_type="grayscale")
    eval_env = gym.wrappers.AtariPreprocessing(
        eval_env, frame_skip=4, grayscale_obs=True,
        screen_size=84, scale_obs=False
    )
    eval_env = gym.wrappers.FrameStack(eval_env, 4)
    n_actions = eval_env.action_space.n
    logging.info(f"Eval env ready — n_actions={n_actions}")

    h5 = h5py.File(dataset_path, "r")
    obs_ds  = h5["observations"] 
    act_ds  = h5["actions"]
    rew_ds  = h5["rewards"]
    term_ds = h5["terminals"]
    N = len(obs_ds)
    logging.info(f"Loaded dataset with {N} transitions")

    sample = obs_ds[0]
    if sample.ndim == 3 and sample.shape[0] in [1,4]:
        in_ch = sample.shape[0]
    elif sample.ndim == 3 and sample.shape[-1] in [1,4]:
        in_ch = sample.shape[-1]
    else:
        in_ch = 1
    logging.info(f"Inferred in_channels={in_ch}")

    all_rews = np.array(rew_ds[:])
    reward_idxs = np.where(all_rews != 0)[0]
    logging.info(f"Found {len(reward_idxs)} ±1 transitions for oversampling")

    dev = torch.device(device)
    policy_net = DQNNetwork(in_ch, n_actions).to(dev)
    target_net = DQNNetwork(in_ch, n_actions).to(dev)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-4)

    gamma = 0.99
    target_update_freq = 8000

    logs = [("step", "eval/average_reward")]
    t_start = time.time()

    pbar = tqdm(range(1, steps+1), desc="DQN Training", ncols=100)
    for step in pbar:
        num_special = max(1, batch_size // 4)
        num_uniform = batch_size - num_special
        idx_uniform = np.random.randint(0, N-1, size=num_uniform)
        pool = reward_idxs[reward_idxs < N-1]
        if len(pool) >= num_special:
            idx_special = np.random.choice(pool, num_special, replace=False)
        else:
            idx_special = np.random.choice(pool, num_special, replace=True)
        idxs = np.concatenate([idx_uniform, idx_special])
        np.random.shuffle(idxs)

        obs_batch = np.stack([
            np.transpose(obs_ds[i], (2,0,1)) if obs_ds[i].ndim==3 and obs_ds[i].shape[-1] in [1,4]
            else obs_ds[i]
            for i in idxs
        ], axis=0)
        nxt_batch = np.stack([
            np.transpose(obs_ds[i+1], (2,0,1)) if obs_ds[i+1].ndim==3 and obs_ds[i+1].shape[-1] in [1,4]
            else obs_ds[i+1]
            for i in idxs
        ], axis=0)
        acts  = np.array([act_ds[i] for i in idxs], dtype=np.int64)
        rews  = np.array([rew_ds[i] for i in idxs], dtype=np.float32)
        dones = np.array([term_ds[i] for i in idxs], dtype=np.float32)

        obs_t  = torch.tensor(obs_batch, dtype=torch.float32, device=dev)
        nxt_t  = torch.tensor(nxt_batch, dtype=torch.float32, device=dev)
        acts_t = torch.tensor(acts, dtype=torch.long, device=dev)
        rews_t = torch.tensor(rews, dtype=torch.float32, device=dev)
        done_t = torch.tensor(dones, dtype=torch.float32, device=dev)
        
        rews_t = torch.clamp(rews_t, -1.0, 1.0)

        q_vals = policy_net(obs_t)
        q_sa   = q_vals.gather(1, acts_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            q_next_policy = policy_net(nxt_t)
            next_actions  = q_next_policy.argmax(dim=1, keepdim=True)
            q_next_target = target_net(nxt_t).gather(1, next_actions).squeeze(1)
            q_target      = rews_t + gamma * q_next_target * (1 - done_t)
            
            td_error = q_target - q_sa
            td_error = torch.clamp(td_error, -td_clip, td_clip)
            q_target = q_sa + td_error
        
        td_loss = F.smooth_l1_loss(q_sa, q_target)
        
        if use_cql:
            logsumexp_q = torch.logsumexp(q_vals, dim=1)
                       
            cql_loss = (logsumexp_q - q_sa).mean()
            
            
            cql_loss = torch.clamp(cql_loss, -20.0, 20.0) 
            loss = td_loss + cql_alpha * cql_loss
            
            loss = torch.clamp(loss, -100.0, 100.0)
        else:
            loss = td_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())
            logging.info(f"[Step {step}] Target network synced")

        elapsed = time.time() - t_start
        fps = step / elapsed
        gpu_mem = ""
        if dev.type == "cuda":
            alloc = torch.cuda.memory_allocated(dev) / 1e9
            resv  = torch.cuda.memory_reserved(dev) / 1e9
            gpu_mem = f", GPU mem {alloc:.2f}G/{resv:.2f}G"
        nz_frac = (rews_t != 0).float().mean().item()
        pbar.set_postfix({
            "loss":  f"{loss.item():.3f}",
            "fps":   f"{fps:.1f}{gpu_mem}",
            "nz/bs": f"{int(nz_frac*batch_size)}/{batch_size}"
        })

        avg_q = q_vals.mean().item()
        
        if step % 20_000 == 0:
            logging.info(f"[Step {step}] Avg Q-value: {avg_q:.3f}, Loss: {loss.item():.3f}")
            
        if step % 50_000 == 0 or step == steps:  
            total_r = 0.0
            trials = 5
            for _ in range(trials):
                o, _ = eval_env.reset()
                done_e = False
                ep_r = 0.0
                while not done_e:
                    o_np = np.array(o)
                    if o_np.ndim==3 and o_np.shape[-1] in [1,4]:
                        o_np = np.transpose(o_np, (2,0,1))
                    o_t = torch.tensor(o_np, dtype=torch.float32, device=dev).unsqueeze(0)
                    with torch.no_grad():
                        qs = policy_net(o_t)
                        a  = int(qs.argmax(dim=1).item())
                    o, r, done_e, _, _ = eval_env.step(a)
                    ep_r += r
                total_r += ep_r
            avg_r = total_r / trials
            logging.info(f"[Step {step}] Eval avg return = {avg_r:.2f}")
            logs.append((step, avg_r))

    torch.save(policy_net.state_dict(), "dqn_pong.pt")
    with open("dqn_logs.csv", "w") as f:
        f.write("step,eval/average_reward\n")
        for s, val in logs[1:]:
            f.write(f"{s},{val:.2f}\n")

    h5.close()
    logging.info("Training complete. Model and logs saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    default="pong_offline.h5")
    parser.add_argument("--steps",      type=int,   default=500_000)
    parser.add_argument("--batch-size", type=int,   default=128) 
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--cql-alpha", type=float, default=1.0,
                        help="Weight for CQL loss term (Conservative Q-Learning)")
    parser.add_argument("--use-cql",   action="store_true", default=True,
                        help="Whether to use Conservative Q-Learning")
    parser.add_argument("--td-clip",   type=float, default=10.0,
                        help="Clipping threshold for TD errors")
    args = parser.parse_args()
    train(args.dataset, args.steps, args.batch_size, args.device, args.seed, 
          args.cql_alpha, args.use_cql, args.td_clip)
