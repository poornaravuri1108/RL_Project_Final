#!/usr/bin/env python
import argparse, gymnasium as gym, torch, numpy as np
from d3rlpy.algos import DQN
from collections import deque

def eval_dqn(model_path, episodes=100, render=False):
    env = gym.make("ALE/Pong-v5", obs_type="grayscale", frameskip=4,
                   render_mode="human" if render else None)
    model = DQN.load_model(model_path)
    scores = []
    for _ in range(episodes):
        done, score, obs = False, 0, env.reset()[0]
        while not done:
            act = model.predict([obs])[0]
            obs, rew, done, _, _ = env.step(act)
            score += rew
        scores.append(score)
    print(f"DQN avg score over {episodes} eps: {np.mean(scores):.2f}")

def eval_ppo(model_path, episodes=100, render=False):
    env = gym.make("ALE/Pong-v5", obs_type="grayscale", frameskip=4,
                   render_mode="human" if render else None)
    from train_offline_ppo import ActorCritic
    net = ActorCritic(n_actions=env.action_space.n)
    net.load_state_dict(torch.load(model_path))
    net.eval()
    scores = []
    for _ in range(episodes):
        done, score, obs = False, 0, env.reset()[0]
        while not done:
            with torch.no_grad():
                logits, _ = net(torch.tensor(obs[None,None],dtype=torch.float32))
                act = torch.argmax(logits, -1).item()
            obs, rew, done, _, _ = env.step(act)
            score += rew
        scores.append(score)
    print(f"PPO avg score over {episodes} eps: {np.mean(scores):.2f}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["dqn","ppo"], required=True)
    p.add_argument("--ckpt", required=True)
    args = p.parse_args()
    (eval_dqn if args.algo=="dqn" else eval_ppo)(args.ckpt)
