#!/usr/bin/env python3
"""
Offline PPO for Atari Pong (minimal action-set)

✓ behaviour-cloning warm-up
✓ batch-wise PPO to avoid CUDA OOM
✓ gradient-norm clip + NaN/Inf guards + advantage clamp
✓ CSV logging of evaluation returns
✓ evaluation env uses AtariPreprocessing + FrameStack → 84×84×4 (same as dataset)

Dataset keys expected:
  observations uint8  (N, H, W, C) or (N, C, H, W)
  actions      uint8/int64 (N,)
  rewards      float32 (N,)
  terminals    bool    (N,)
"""
import argparse, csv, os, sys, h5py, gymnasium as gym
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from gymnasium.wrappers import AtariPreprocessing, FrameStack


# ─────────────────────────── network ────────────────────────────
class ActorCritic(nn.Module):
    def __init__(self, in_ch: int, n_actions: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),   nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),   nn.ReLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_ch, 84, 84)
            conv_out = int(np.prod(self.conv(dummy).shape[1:]))
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(conv_out, 512), nn.ReLU())
        self.pi = nn.Linear(512, n_actions)
        self.v  = nn.Linear(512, 1)

    def forward(self, x):
        x = x.float() / 255.0
        x = self.fc(self.conv(x))
        return self.pi(x), self.v(x).squeeze(-1)


# ─────────────────────── helper functions ───────────────────────
def to_chw(arr: np.ndarray) -> np.ndarray:
    """Convert NHWC/HWC/HW to NCHW/CHW."""
    if arr.ndim == 4:
        return arr if arr.shape[1] in (1, 4) else arr.transpose(0, 3, 1, 2)
    if arr.ndim == 3:
        if arr.shape[0] in (1, 4):
            return arr
        if arr.shape[2] in (1, 4):
            return arr.transpose(2, 0, 1)
        return arr[None, ...]
    if arr.ndim == 2:
        return arr[None, None, ...]
    raise ValueError(f"Unexpected obs shape {arr.shape}")


def preprocess_obs(np_obs) -> torch.Tensor:
    return torch.from_numpy(to_chw(np_obs))


def discount_cumsum(rew, done, gamma):
    out, csum = np.zeros_like(rew), 0.0
    for i in reversed(range(len(rew))):
        csum = rew[i] + gamma * csum * (1.0 - done[i])
        out[i] = csum
    return out


# ───────────── behaviour-cloning warm-up ─────────────
def behaviour_cloning(net, opt, obs, act, bs, epochs, device):
    ce = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(obs, act), bs, shuffle=True, drop_last=True)
    net.train()
    for ep in range(1, epochs + 1):
        tot = 0.0
        for o, a in loader:
            o, a = o.to(device), a.to(device)
            loss = ce(net(o)[0], a)
            opt.zero_grad(set_to_none=True)
            loss.backward(); opt.step()
            tot += loss.item() * len(o)
        print(f"[BC] epoch {ep}/{epochs} | loss {tot/len(obs):.4f}")


# ──────────────────── PPO minibatch update ────────────────────
def ppo_update(net, opt, obs, act, old_lp, ret, adv,
               clip, vf_coef, ent_coef, gd_epochs, mb_size,
               grad_clip, device, kl_coef=0.5, bc_coef=0.2):
    dataset = TensorDataset(obs, act, old_lp, ret, adv)
    loader = DataLoader(dataset, batch_size=mb_size, shuffle=True, drop_last=True)
    for _ in range(gd_epochs):
        for o, a, olp, r, ad in loader:
            o, a, olp, r, ad = [t.to(device) for t in (o, a, olp, r, ad)]

            logits, v = net(o)
            if not torch.isfinite(logits).all() or not torch.isfinite(v).all():
                continue  # skip corrupt batch

            dist = torch.distributions.Categorical(logits=logits)
            lp = dist.log_prob(a)
            ratio = torch.exp(lp - olp)
            if not torch.isfinite(ratio).all():
                continue

            obj1 = ratio * ad
            obj2 = torch.clamp(ratio, 1 - clip, 1 + clip) * ad
            # Standard PPO objective
            loss_pi = -torch.min(obj1, obj2).mean()
            
            # Value loss
            loss_v = F.mse_loss(v, r)
            
            # Behavior cloning loss to stay close to dataset policy
            bc_loss = F.cross_entropy(logits, a)
            
            # KL divergence term to prevent policy deviation
            kl_div = (torch.exp(lp) * (lp - olp)).mean()
            
            # Combined loss
            loss = loss_pi + vf_coef * loss_v + kl_coef * kl_div + bc_coef * bc_loss - ent_coef * dist.entropy().mean()
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
            opt.step()


# ── Simple evaluation function for testing during training ──
def evaluate_policy(net, env, device, n_episodes=3):
    net.eval()
    total_reward = 0.0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            tensor = preprocess_obs(np.asarray(obs)).unsqueeze(0).to(device)
            with torch.no_grad():
                act = torch.argmax(net(tensor)[0]).item()
            obs, r, term, trunc, _ = env.step(act)
            total_reward += r
            done = term or trunc
    return total_reward / n_episodes

# ─────────────────────────── main ────────────────────────────
def main():
    pa = argparse.ArgumentParser()
    pa.add_argument('--dataset', required=True)
    pa.add_argument('--epochs', type=int, default=30)
    pa.add_argument('--bc_epochs', type=int, default=5)  # Increased BC epochs
    pa.add_argument('--batch_size', type=int, default=4096)
    pa.add_argument('--ppo_mb', type=int, default=256)
    pa.add_argument('--ppo_gd', type=int, default=4)
    pa.add_argument('--clip', type=float, default=0.2)
    pa.add_argument('--vf_coef', type=float, default=0.5)
    pa.add_argument('--ent_coef', type=float, default=0.005)  # Reduced entropy coefficient
    pa.add_argument('--gamma', type=float, default=0.99)
    pa.add_argument('--lam',   type=float, default=0.95)
    pa.add_argument('--lr',    type=float, default=2.5e-5)  # Even lower learning rate for stability
    pa.add_argument('--grad_clip', type=float, default=0.5)
    pa.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = pa.parse_args()
    device = torch.device(args.device)

    # ─── evaluation env (minimal 6-action set) ───
    base = gym.make("ALE/Pong-v5", frameskip=1,
                    full_action_space=False, render_mode=None)
    env = FrameStack(AtariPreprocessing(base, grayscale_obs=True,
                                        scale_obs=True, frame_skip=1), 4)
    n_actions = env.action_space.n

    # ─── load dataset ───
    with h5py.File(args.dataset, 'r') as h5:
        obs_np  = h5['observations'][:]
        act_np  = h5['actions'][:].astype(np.int64)
        rew_np  = h5['rewards'][:]
        done_np = h5['terminals'][:].astype(bool)

    if act_np.max()+1 != n_actions:
        sys.exit(f"Dataset actions ({act_np.max()+1}) ≠ env actions ({n_actions})")

    obs_t = preprocess_obs(obs_np)          # uint8 CHW
    act_t = torch.from_numpy(act_np).long()
    ret_t = torch.from_numpy(discount_cumsum(rew_np, done_np, args.gamma).astype(np.float32))

    # ─── model & optimiser ───
    net = ActorCritic(obs_t.shape[1], n_actions).to(device)
    # Use a different optimizer with weight decay for regularization
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)

    # ─── behaviour cloning ───
    if args.bc_epochs:
        behaviour_cloning(net, opt, obs_t, act_t,
                          bs=args.ppo_mb, epochs=args.bc_epochs, device=device)

    # ─── CSV logger ───
    log_path = os.path.splitext(args.dataset)[0] + "_training_log.csv"
    with open(log_path, 'w', newline='') as fcsv:
        writer = csv.writer(fcsv); writer.writerow(['epoch', 'eval_return'])

        # ─── Add checkpoint saver ───
        best_return = float('-inf')
        best_epoch = 0
        
        # ─── PPO epochs ───
        for ep in range(1, args.epochs + 1):
            # --- compute old log-probs + values batch-wise ---
            net.eval()
            olp_list, v_list = [], []
            loader = DataLoader(TensorDataset(obs_t, act_t),
                                batch_size=args.batch_size, shuffle=False)
            with torch.no_grad():
                for o_b, a_b in loader:
                    logits, v = net(o_b.to(device))
                    dist = torch.distributions.Categorical(logits=logits)
                    olp_list.append(dist.log_prob(a_b.to(device)).cpu())
                    v_list.append(v.cpu())
            old_lp = torch.cat(olp_list)
            val_t  = torch.cat(v_list)

            adv_t = (ret_t - val_t)
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
            adv_t = torch.clamp(adv_t, -10.0, 10.0)  # clamp extremes

            # --- PPO update ---
            net.train()
            ppo_update(net, opt, obs_t, act_t, old_lp, ret_t, adv_t,
                       clip=args.clip, vf_coef=args.vf_coef, ent_coef=args.ent_coef,
                       gd_epochs=args.ppo_gd, mb_size=args.ppo_mb,
                       grad_clip=args.grad_clip, device=device,
                       kl_coef=0.5, bc_coef=0.2)  # Add KL and BC regularization

            # --- quick evaluation ---
            net.eval()
            obs, _ = env.reset(); ep_ret, done = 0.0, False
            while not done:
                tensor = preprocess_obs(np.asarray(obs)).unsqueeze(0).to(device)
                with torch.no_grad():
                    act = torch.argmax(net(tensor)[0]).item()
                obs, r, term, trunc, _ = env.step(act)
                ep_ret += r; done = term or trunc

            # Save model if it's the best so far
            if ep_ret > best_return:
                best_return = ep_ret
                best_epoch = ep
                torch.save(net.state_dict(), os.path.splitext(args.dataset)[0] + "_best_ppo.pt")
                print(f"Epoch {ep:03d}/{args.epochs} | eval return {ep_ret:5.1f} | NEW BEST!")
            else:
                print(f"Epoch {ep:03d}/{args.epochs} | eval return {ep_ret:5.1f} | Best: {best_return:.1f} (ep {best_epoch})")
            
            writer.writerow([ep, ep_ret]); fcsv.flush()
            
            # Simple early stopping - if we're close to winning, do more evaluation to confirm
            if ep_ret > 15:
                confirm_return = evaluate_policy(net, env, device, n_episodes=5)
                print(f"Confirmation evaluation: {confirm_return:.1f}")
                if confirm_return > 18:  # If consistently good, we can stop
                    print(f"Early stopping at epoch {ep} with return {confirm_return:.1f}")
                    break

    env.close()
    ckpt_path = os.path.splitext(args.dataset)[0] + "_ppo.pt"
    torch.save(net.state_dict(), ckpt_path)
    print(f"✓ finished - model → {ckpt_path}\n✓ log → {log_path}")


if __name__ == "__main__":
    main()

