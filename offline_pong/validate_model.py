#!/usr/bin/env python3
"""
validate_model.py

Quick validation utility for DQN and PPO models - runs a few episodes
and reports average score to check if model quality is improving.

Usage:
    python validate_model.py --model dqn_pong.pt --type dqn
    python validate_model.py --model pong_offline_best_ppo.pt --type ppo
"""

import argparse
import numpy as np
import torch
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStack
from tqdm import tqdm

# Import model classes from training files
from train_dqn import DQNNetwork
from train_offline_ppo import ActorCritic, preprocess_obs


def evaluate_dqn(model_path, device, episodes=10, render=False):
    # Create environment
    render_mode = "human" if render else None
    env = gym.make("ALE/Pong-v5", frameskip=1, obs_type="grayscale", render_mode=render_mode)
    env = AtariPreprocessing(env, frame_skip=4, grayscale_obs=True, scale_obs=False, screen_size=84)
    env = FrameStack(env, 4)
    n_actions = env.action_space.n
    
    # Load model
    model = DQNNetwork(4, n_actions).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Run evaluation
    total_rewards = []
    with torch.no_grad():
        for episode in tqdm(range(episodes), desc="Evaluating DQN"):
            obs, _ = env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                # Convert observation
                obs_np = np.array(obs)
                if obs_np.ndim == 3 and obs_np.shape[-1] in [1, 4]:
                    obs_np = np.transpose(obs_np, (2, 0, 1))
                
                # Get action
                obs_tensor = torch.tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
                q_values = model(obs_tensor)
                action = int(q_values.argmax(dim=1).item())
                
                # Step environment
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                done = terminated or truncated
            
            total_rewards.append(episode_reward)
    
    env.close()
    return total_rewards


def evaluate_ppo(model_path, device, episodes=10, render=False):
    # Create environment
    render_mode = "human" if render else None
    base = gym.make("ALE/Pong-v5", frameskip=1, full_action_space=False, render_mode=render_mode)
    env = FrameStack(AtariPreprocessing(base, grayscale_obs=True, scale_obs=True, frame_skip=1), 4)
    n_actions = env.action_space.n
    
    # Load model
    model = ActorCritic(4, n_actions).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Run evaluation
    total_rewards = []
    with torch.no_grad():
        for episode in tqdm(range(episodes), desc="Evaluating PPO"):
            obs, _ = env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                # Get action
                tensor = preprocess_obs(np.asarray(obs)).unsqueeze(0).to(device)
                logits, _ = model(tensor)
                action = torch.argmax(logits).item()
                
                # Step environment
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                done = terminated or truncated
            
            total_rewards.append(episode_reward)
    
    env.close()
    return total_rewards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to the model file")
    parser.add_argument("--type", choices=["dqn", "ppo"], required=True, help="Model type (dqn or ppo)")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to evaluate")
    parser.add_argument("--render", action="store_true", help="Render the environment")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    # Evaluate the model
    if args.type == "dqn":
        rewards = evaluate_dqn(args.model, device, args.episodes, args.render)
    else:
        rewards = evaluate_ppo(args.model, device, args.episodes, args.render)
    
    # Print results
    print(f"\nEvaluation results for {args.model} ({args.type}):")
    print(f"  Episodes: {args.episodes}")
    print(f"  Average reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"  Min/Max reward: {np.min(rewards):.1f}/{np.max(rewards):.1f}")
    print(f"  Rewards: {rewards}")


if __name__ == "__main__":
    main()
