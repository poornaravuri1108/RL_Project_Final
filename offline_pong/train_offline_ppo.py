#!/usr/bin/env python
"""
Batch-mode PPO on fixed trajectories (no env interaction).
Caveat: performance is usually weaker than DQN in pure offline settings.
"""
import argparse, collections, torch, numpy as np
from torch import nn
from torch.utils.data import DataLoader
from d3rlpy.dataset import MDPDataset
from tqdm import tqdm

class CNNBackbone(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(1, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32,64,4,2),   nn.ReLU(),
            nn.Conv2d(64,64,3,1),   nn.ReLU(),
            nn.Flatten(),
        )
        self.output_dim = 7*7*64

class ActorCritic(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.body = CNNBackbone()
        self.policy = nn.Sequential(
            nn.Linear(self.body.output_dim, 512), nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        self.value = nn.Sequential(
            nn.Linear(self.body.output_dim, 512), nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        z = self.body(x / 255.0)      # keep raw uint8 images
        return self.policy(z), self.value(z).squeeze(-1)

def make_batches(dataset, batch_size=64):
    """Yield traj-level batches to respect episode boundaries."""
    Episode = collections.namedtuple("Episode", "obs act rew done")
    # gather transitions per episode
    episodes, cur = [], Episode([],[],[],[])
    for o,a,r,t in zip(dataset.observations, dataset.actions,
                       dataset.rewards, dataset.terminals):
        cur.obs.append(o); cur.act.append(a); cur.rew.append(r); cur.done.append(t)
        if t:
            episodes.append(Episode(*map(np.asarray,(cur.obs,cur.act,cur.rew,cur.done))))
            cur = Episode([],[],[],[])
    # flatten into sequences
    for ep in episodes:
        obs  = torch.tensor(ep.obs, dtype=torch.float32).unsqueeze(1)
        act  = torch.tensor(ep.act, dtype=torch.long)
        rew  = torch.tensor(ep.rew, dtype=torch.float32)
        yield obs, act, rew                 # each element is (T, …)

def compute_returns(rew, gamma=0.99):
    ret = torch.zeros_like(rew)
    g   = 0.0
    for t in reversed(range(len(rew))):
        g = rew[t] + gamma * g
        ret[t] = g
    return ret

def train(dataset_path, epochs=10, clip=0.2, lr=2.5e-4,
          minibatch=256, update_epochs=4, seed=42):
    torch.manual_seed(seed)
    dataset = MDPDataset.load(dataset_path)
    n_actions = int(dataset.actions.max() + 1)
    net = ActorCritic(n_actions)
    optim = torch.optim.Adam(net.parameters(), lr=lr)

    for epoch in range(epochs):
        losses, returns = [], []
        for obs, act, rew in make_batches(dataset):
            with torch.no_grad():
                logit, val = net(obs)
                dist = torch.distributions.Categorical(logits=logit)
                old_logp = dist.log_prob(act)
                adv  = compute_returns(rew) - val             # crude advantage

            # PPO inner updates
            for _ in range(update_epochs):
                idx = torch.randperm(len(obs))
                for start in range(0, len(obs), minibatch):
                    sl = idx[start:start+minibatch]
                    l_obs, l_act, l_adv, l_ret, l_old = obs[sl], act[sl], adv[sl], compute_returns(rew)[sl], old_logp[sl]

                    logit, val = net(l_obs)
                    dist  = torch.distributions.Categorical(logits=logit)
                    logp  = dist.log_prob(l_act)
                    ratio = torch.exp(logp - l_old)

                    # clipped surrogate
                    surr = torch.min(ratio * l_adv,
                                     torch.clamp(ratio, 1-clip, 1+clip) * l_adv).mean()
                    v_loss = (l_ret - val).pow(2).mean()
                    entropy = dist.entropy().mean()
                    loss = -surr + 0.5*v_loss - 0.01*entropy

                    optim.zero_grad(); loss.backward(); optim.step()
                    losses.append(loss.item())
        print(f"[epoch {epoch+1}] loss={np.mean(losses):.4f}")

    torch.save(net.state_dict(), "ppo_offline.pt")

if __name__ == "__main__":
    import argparse, os
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="pong_offline.h5")
    p.add_argument("--epochs",  type=int, default=10)
    train(**vars(p.parse_args()))
