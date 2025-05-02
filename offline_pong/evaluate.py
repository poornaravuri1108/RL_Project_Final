import argparse
import gymnasium as gym
import numpy as np
import torch

from train_dqn import DQNNetwork
from train_offline_ppo import ActorCritic

def evaluate_dqn(model_path: str, episodes: int = 100, render: bool = False):
    # Create environment
    env = gym.make("ALE/Pong-v5", frameskip=1, obs_type="grayscale",
                   render_mode="human" if render else None)
    env = gym.wrappers.AtariPreprocessing(env, frame_skip=4, grayscale_obs=True,
                                          grayscale_newaxis=False, screen_size=84, scale_obs=False)
    env = gym.wrappers.FrameStack(env, 4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Determine obs channels and actions from env
    obs_shape = env.observation_space.shape  # e.g. (84,84,4)
    if len(obs_shape) == 3:
        # shape could be (H,W,C)
        in_channels = obs_shape[-1] if obs_shape[-1] in [1, 4] else obs_shape[0]
    else:
        in_channels = 1
    n_actions = env.action_space.n
    # Load DQN model
    model = DQNNetwork(in_channels, n_actions)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    scores = []
    for ep in range(episodes):
        obs, _ = env.reset()
        obs = np.array(obs)
        done = False
        score = 0.0
        while not done:
            # Prepare observation
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
            if obs_tensor.ndim == 3 and obs_tensor.shape[0] not in [1, 4]:
                obs_tensor = obs_tensor.permute(2, 0, 1)
            elif obs_tensor.ndim == 2:
                obs_tensor = obs_tensor.unsqueeze(0)
            with torch.no_grad():
                q_values = model(obs_tensor.unsqueeze(0))
                action = int(torch.argmax(q_values, dim=1).item())
            obs, reward, done, _, _ = env.step(action)
            obs = np.array(obs)
            score += reward
            if render:
                env.render()
        scores.append(score)
    avg_score = np.mean(scores)
    print(f"DQN average score over {episodes} episodes: {avg_score:.2f}")

def evaluate_ppo(model_path: str, episodes: int = 100, render: bool = False):
    env = gym.make("ALE/Pong-v5", frameskip=1, obs_type="grayscale",
                   render_mode="human" if render else None)
    env = gym.wrappers.AtariPreprocessing(env, frame_skip=4, grayscale_obs=True,
                                          grayscale_newaxis=False, screen_size=84, scale_obs=False)
    env = gym.wrappers.FrameStack(env, 4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_actions = env.action_space.n
    # Determine channels from env observation space
    obs_shape = env.observation_space.shape  # (84,84,4) likely
    if len(obs_shape) == 3:
        in_channels = obs_shape[-1] if obs_shape[-1] in [1, 4] else obs_shape[0]
    else:
        in_channels = 1
    # Load PPO model
    net = ActorCritic(in_channels, n_actions)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.to(device)
    net.eval()
    scores = []
    for ep in range(episodes):
        obs, _ = env.reset()
        obs = np.array(obs)
        done = False
        score = 0.0
        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
            if obs_tensor.ndim == 3 and obs_tensor.shape[0] not in [1, 4]:
                obs_tensor = obs_tensor.permute(2, 0, 1)
            elif obs_tensor.ndim == 2:
                obs_tensor = obs_tensor.unsqueeze(0)
            with torch.no_grad():
                logits, _ = net(obs_tensor.unsqueeze(0))
                action = int(torch.argmax(logits, dim=-1).item())
            obs, reward, done, _, _ = env.step(action)
            obs = np.array(obs)
            score += reward
            if render:
                env.render()
        scores.append(score)
    avg_score = np.mean(scores)
    print(f"PPO average score over {episodes} episodes: {avg_score:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["dqn", "ppo"], required=True, help="Which algorithm to evaluate")
    parser.add_argument("--ckpt", required=True, help="Path to the saved model checkpoint")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes to evaluate over")
    parser.add_argument("--render", action="store_true", help="Render the environment during evaluation")
    args = parser.parse_args()
    if args.algo == "dqn":
        evaluate_dqn(args.ckpt, episodes=args.episodes, render=args.render)
    else:
        evaluate_ppo(args.ckpt, episodes=args.episodes, render=args.render)
