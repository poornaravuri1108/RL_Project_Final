#!/usr/bin/env python3
"""
Behavior-Cloning with DQN for Atari Pong - Hybrid approach

Combines the best elements of our successful DQN approach with behavior cloning:
✓ Q-learning for value estimation
✓ Behavior cloning for policy initialization
✓ Conservative regularization similar to CQL
✓ Double Q-learning for stable updates

Usage:
    python train_bc_dqn.py --dataset pong_offline_diverse.h5 --steps 500000 --device cuda
"""
import argparse
import logging
import time
import os

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
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 84, 84)
            conv_out = self.conv(dummy).shape[1]
        
        # Dueling architecture
        self.value_stream = nn.Sequential(
            nn.Linear(conv_out, 512), nn.ReLU(), 
            nn.Dropout(0.1),  # Add dropout for regularization
            nn.Linear(512, 1)
        )
        self.adv_stream = nn.Sequential(
            nn.Linear(conv_out, 512), nn.ReLU(), 
            nn.Dropout(0.1),  # Add dropout for regularization
            nn.Linear(512, n_actions)
        )

    def forward(self, x):
        x = x / 255.0
        feat = self.conv(x)
        val  = self.value_stream(feat)
        adv  = self.adv_stream(feat)
        return val + adv - adv.mean(dim=1, keepdim=True)


def behavior_cloning_pretraining(policy_net, optimizer, obs_ds, act_ds, batch_size, epochs, device):
    """Pre-train the policy network with behavior cloning"""
    logging.info(f"Starting behavior cloning pre-training for {epochs} epochs...")
    criterion = nn.CrossEntropyLoss()
    N = len(obs_ds)
    
    for epoch in range(1, epochs+1):
        total_loss = 0.0
        batch_count = 0
        indices = np.random.permutation(N)
        
        for start_idx in range(0, N - batch_size + 1, batch_size):
            # Get batch indices
            batch_indices = indices[start_idx:start_idx + batch_size]
            
            # Extract batch data
            obs_batch = np.stack([
                np.transpose(obs_ds[i], (2,0,1)) if obs_ds[i].ndim==3 and obs_ds[i].shape[-1] in [1,4]
                else obs_ds[i]
                for i in batch_indices
            ], axis=0)
            acts = np.array([act_ds[i] for i in batch_indices], dtype=np.int64)
            
            # To tensors
            obs_t = torch.tensor(obs_batch, dtype=torch.float32, device=device)
            acts_t = torch.tensor(acts, dtype=torch.long, device=device)
            
            # Forward pass
            q_vals = policy_net(obs_t)
            loss = criterion(q_vals, acts_t)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            
        avg_loss = total_loss / batch_count
        logging.info(f"[BC] Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f}")


def random_crop(x, padding=4):
    """Apply random crop data augmentation to batch of observations."""
    b, c, h, w = x.shape
    padded = F.pad(x, (padding, padding, padding, padding), mode='constant', value=0)
    crops = []
    for i in range(b):
        top = np.random.randint(0, padding * 2)
        left = np.random.randint(0, padding * 2)
        crops.append(padded[i:i+1, :, top:top+h, left:left+w])
    return torch.cat(crops, dim=0)


def train(dataset_path, steps, batch_size, device, seed, 
          bc_epochs=10, cql_alpha=2.0, use_cql=True, td_clip=10.0,
          use_augmentation=True):
    # Configure logging
    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(message)s",
        level=logging.INFO
    )
    logging.info(f"Starting Hybrid BC-DQN training: dataset={dataset_path}, "
                f"steps={steps}, batch={batch_size}, device={device}")

    # Reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Build eval environment
    eval_env = gym.make("ALE/Pong-v5", frameskip=1, obs_type="grayscale")
    eval_env = gym.wrappers.AtariPreprocessing(
        eval_env, frame_skip=4, grayscale_obs=True,
        screen_size=84, scale_obs=False
    )
    eval_env = gym.wrappers.FrameStack(eval_env, 4)
    n_actions = eval_env.action_space.n
    logging.info(f"Eval env ready — n_actions={n_actions}")

    # Open HDF5 dataset
    h5 = h5py.File(dataset_path, "r")
    obs_ds  = h5["observations"]  # shape (N,4,84,84) or (N,84,84,4)
    act_ds  = h5["actions"]
    rew_ds  = h5["rewards"]
    term_ds = h5["terminals"]
    N = len(obs_ds)
    logging.info(f"Loaded dataset with {N} transitions")

    # Infer input channels
    sample = obs_ds[0]
    if sample.ndim == 3 and sample.shape[0] in [1,4]:
        in_ch = sample.shape[0]
    elif sample.ndim == 3 and sample.shape[-1] in [1,4]:
        in_ch = sample.shape[-1]
    else:
        in_ch = 1
    logging.info(f"Inferred in_channels={in_ch}")

    # Precompute non-zero reward indices
    all_rews = np.array(rew_ds[:])
    reward_idxs = np.where(all_rews != 0)[0]
    logging.info(f"Found {len(reward_idxs)} ±1 transitions for oversampling")

    # Build networks
    dev = torch.device(device)
    policy_net = DQNNetwork(in_ch, n_actions).to(dev)
    target_net = DQNNetwork(in_ch, n_actions).to(dev)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = torch.optim.AdamW(policy_net.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)

    # Behavior cloning pre-training
    if bc_epochs > 0:
        behavior_cloning_pretraining(policy_net, optimizer, obs_ds, act_ds, 
                                     batch_size, bc_epochs, dev)
        # Reset target network after behavior cloning
        target_net.load_state_dict(policy_net.state_dict())

    gamma = 0.99
    target_update_freq = 8000

    # Prepare logs
    logs = [("step", "eval/average_reward", "avg_q_value", "loss")]
    t_start = time.time()
    best_reward = float('-inf')
    best_model_path = os.path.splitext(dataset_path)[0] + "_best_hybrid_dqn.pt"

    # Training loop
    pbar = tqdm(range(1, steps+1), desc="Hybrid BC-DQN Training", ncols=100)
    for step in pbar:
        # Sample 75% uniform, 25% from reward_idxs
        num_special = max(1, batch_size // 4)
        num_uniform = batch_size - num_special
        idx_uniform = np.random.randint(0, N-1, size=num_uniform)
        # Ensure we don't pick the last index for idx+1
        pool = reward_idxs[reward_idxs < N-1]
        if len(pool) >= num_special:
            idx_special = np.random.choice(pool, num_special, replace=False)
        else:
            idx_special = np.random.choice(pool, num_special, replace=True)
        idxs = np.concatenate([idx_uniform, idx_special])
        np.random.shuffle(idxs)

        # Load minibatch and permute channel-last if needed
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

        # To tensors
        obs_t  = torch.tensor(obs_batch, dtype=torch.float32, device=dev)
        nxt_t  = torch.tensor(nxt_batch, dtype=torch.float32, device=dev)
        acts_t = torch.tensor(acts, dtype=torch.long, device=dev)
        rews_t = torch.tensor(rews, dtype=torch.float32, device=dev)
        done_t = torch.tensor(dones, dtype=torch.float32, device=dev)
        
        # Apply data augmentation if enabled
        if use_augmentation and np.random.random() < 0.5:
            obs_t = random_crop(obs_t)
            nxt_t = random_crop(nxt_t)
        
        # Scale rewards for better learning dynamics
        rews_t = torch.clamp(rews_t, -1.0, 1.0)

        # Compute Q(s,a)
        q_vals = policy_net(obs_t)
        q_sa   = q_vals.gather(1, acts_t.unsqueeze(1)).squeeze(1)

        # Compute Double DQN target with TD-error clipping for stability
        with torch.no_grad():
            q_next_policy = policy_net(nxt_t)
            next_actions  = q_next_policy.argmax(dim=1, keepdim=True)
            q_next_target = target_net(nxt_t).gather(1, next_actions).squeeze(1)
            q_target      = rews_t + gamma * q_next_target * (1 - done_t)
            
            # TD-error clipping for stability
            td_error = q_target - q_sa
            td_error = torch.clamp(td_error, -td_clip, td_clip)
            q_target = q_sa + td_error
        
        # Standard TD loss
        td_loss = F.smooth_l1_loss(q_sa, q_target)
        
        # CQL loss for conservative Q-learning if enabled
        if use_cql:
            # Compute logsumexp over Q-values (proper CQL implementation)
            logsumexp_q = torch.logsumexp(q_vals, dim=1)
            
            # CQL specific regularization - more stable implementation
            # This implements min_Q = TD_target - alpha * (logsumexp(Q) - Q(s,a))
            cql_loss = (logsumexp_q - q_sa).mean()
            
            # Apply cql_alpha with proper scaling & clipping to prevent extreme values
            cql_loss = torch.clamp(cql_loss, -20.0, 20.0)  # Prevent extreme values
            loss = td_loss + cql_alpha * cql_loss
            
            # Ensure loss doesn't reach extreme values
            loss = torch.clamp(loss, -100.0, 100.0)
        else:
            loss = td_loss
        
        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=10.0)
        optimizer.step()
        scheduler.step()

        # Sync target network
        if step % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())
            logging.info(f"[Step {step}] Target network synced")

        # Compute performance metrics
        elapsed = time.time() - t_start
        fps = step / elapsed
        gpu_mem = ""
        if dev.type == "cuda":
            alloc = torch.cuda.memory_allocated(dev) / 1e9
            resv  = torch.cuda.memory_reserved(dev) / 1e9
            gpu_mem = f", GPU mem {alloc:.2f}G/{resv:.2f}G"
        # Non-zero reward fraction
        nz_frac = (rews_t != 0).float().mean().item()
        
        # Calculate average Q-values
        avg_q = q_vals.mean().item()
        
        pbar.set_postfix({
            "loss":  f"{loss.item():.3f}",
            "q_val": f"{avg_q:.2f}",
            "fps":   f"{fps:.1f}{gpu_mem}",
            "nz/bs": f"{int(nz_frac*batch_size)}/{batch_size}"
        })
        
        # Add early validation to check progress
        if step % 20_000 == 0:
            logging.info(f"[Step {step}] Avg Q-value: {avg_q:.3f}, Loss: {loss.item():.3f}")
            
        # Periodic evaluation
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
            logs.append((step, avg_r, avg_q, loss.item()))
            
            # Save best model
            if avg_r > best_reward:
                best_reward = avg_r
                torch.save(policy_net.state_dict(), best_model_path)
                logging.info(f"[Step {step}] New best model saved with return {avg_r:.2f}")

    # Save final model and logs
    final_model_path = os.path.splitext(dataset_path)[0] + "_hybrid_dqn.pt"
    torch.save(policy_net.state_dict(), final_model_path)
    with open(os.path.splitext(dataset_path)[0] + "_hybrid_dqn_logs.csv", "w") as f:
        f.write("step,eval/average_reward,avg_q_value,loss\n")
        for s, val, q, l in logs[1:]:
            f.write(f"{s},{val:.2f},{q:.4f},{l:.4f}\n")

    h5.close()
    logging.info(f"Training complete. Models saved to:")
    logging.info(f"  - Best: {best_model_path}")
    logging.info(f"  - Final: {final_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    default="pong_offline_diverse.h5")
    parser.add_argument("--steps",      type=int,   default=500_000)
    parser.add_argument("--batch-size", type=int,   default=128)
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--bc-epochs",  type=int,   default=10,
                       help="Number of behavior cloning pre-training epochs")
    parser.add_argument("--cql-alpha",  type=float, default=2.0,
                       help="Weight for CQL loss term (Conservative Q-Learning)")
    parser.add_argument("--use-cql",    action="store_true", default=True,
                       help="Whether to use Conservative Q-Learning")
    parser.add_argument("--td-clip",    type=float, default=10.0,
                       help="Clipping threshold for TD errors")
    parser.add_argument("--use-augmentation", action="store_true", default=True,
                       help="Whether to use data augmentation")
    args = parser.parse_args()
    train(args.dataset, args.steps, args.batch_size, args.device, args.seed,
          args.bc_epochs, args.cql_alpha, args.use_cql, args.td_clip, 
          args.use_augmentation)
