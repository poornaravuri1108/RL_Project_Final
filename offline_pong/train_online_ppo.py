import argparse
import csv
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStack
from gymnasium.vector import SyncVectorEnv
from tqdm import tqdm


class ActorCritic(nn.Module):
    def __init__(self, in_ch: int, n_actions: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_ch, 84, 84)
            conv_out = int(np.prod(self.conv(dummy).shape[1:]))
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out, 512), nn.ReLU(),
        )
        self.pi = nn.Linear(512, n_actions)
        self.v = nn.Linear(512, 1)

    def forward(self, x):
        x = x.float() / 255.0
        x = self.fc(self.conv(x))
        return self.pi(x), self.v(x).squeeze(-1)


def to_tensor(obs_np, device):
    if obs_np.shape[-1] in (1, 4):  
        obs_np = np.transpose(obs_np, (0, 3, 1, 2))
    return torch.from_numpy(obs_np).to(device)


# GAE calculation
def compute_gae(rewards, values, dones, next_values, gamma=0.99, lam=0.95):
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = next_values[t]
        else:
            next_value = values[t + 1]
        
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        
        advantages[t] = last_gae = delta + gamma * lam * (1 - dones[t]) * last_gae
    
    returns = advantages + values
    
    return returns, advantages


# PPO update
def ppo_update(model, optimizer, states, actions, old_log_probs, returns, advantages,
               clip_ratio=0.1, vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5,
               update_epochs=4, batch_size=64, device='cuda'):
    states = torch.FloatTensor(states).to(device)
    actions = torch.LongTensor(actions).to(device)
    old_log_probs = torch.FloatTensor(old_log_probs).to(device)
    returns = torch.FloatTensor(returns).to(device)
    advantages = torch.FloatTensor(advantages).to(device)
    
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    dataset = TensorDataset(states, actions, old_log_probs, returns, advantages)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for _ in range(update_epochs):
        for batch in loader:
            b_states, b_actions, b_old_log_probs, b_returns, b_advantages = batch
            
            logits, values = model(b_states)
            
            dist = torch.distributions.Categorical(logits=logits)
            new_log_probs = dist.log_prob(b_actions)
            entropy = dist.entropy().mean()
            
            ratio = torch.exp(new_log_probs - b_old_log_probs)
            clip_adv = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * b_advantages
            policy_loss = -torch.min(ratio * b_advantages, clip_adv).mean()
            
            # Value loss
            value_loss = F.mse_loss(values, b_returns)
            
            # Total loss
            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy
            
            # Optimize
            optimizer.zero_grad()
            loss.backward()
            # Clip gradients
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()


class PongRewardShaping(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev_obs = None
        self.ball_position = None
        self.paddle_position = None
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_obs = obs
        self.ball_position = None
        self.paddle_position = None
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        shaped_reward = reward
        
        if not terminated and reward == 0:
            shaped_reward += 0.01
        
        if reward > 0:
            shaped_reward = 2.0  
            
        self.prev_obs = obs
        
        return obs, shaped_reward, terminated, truncated, info


def make_env(seed=0, difficulty=0):
    def _init():
        env = gym.make("ALE/Pong-v5", frameskip=1, full_action_space=False, difficulty=difficulty)
        env = AtariPreprocessing(env, grayscale_obs=True, scale_obs=False, frame_skip=4)
        env = FrameStack(env, 4)
        env = PongRewardShaping(env)  
        env.seed(seed)
        return env
    return _init


def collect_rollout(envs, model, rollout_steps, device):
    """Collect a batch of rollout data from vectorized environments"""
    num_envs = envs.num_envs
    
    states = np.zeros((rollout_steps, num_envs, 4, 84, 84), dtype=np.uint8)
    actions = np.zeros((rollout_steps, num_envs), dtype=np.int64)
    rewards = np.zeros((rollout_steps, num_envs), dtype=np.float32)
    dones = np.zeros((rollout_steps, num_envs), dtype=np.bool_)
    values = np.zeros((rollout_steps, num_envs), dtype=np.float32)
    log_probs = np.zeros((rollout_steps, num_envs), dtype=np.float32)
    
    obs, _ = envs.reset()
    
    for t in range(rollout_steps):
        obs_tensor = to_tensor(obs, device)
        
        with torch.no_grad():
            logits, value = model(obs_tensor)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        
        # Step environments
        next_obs, reward, term, trunc, _ = envs.step(action.cpu().numpy())
        done = np.logical_or(term, trunc)
        
        # Store data
        states[t] = obs
        actions[t] = action.cpu().numpy()
        rewards[t] = reward
        dones[t] = done
        values[t] = value.cpu().numpy()
        log_probs[t] = log_prob.cpu().numpy()
        
        # Update observation
        obs = next_obs
    
    with torch.no_grad():
        obs_tensor = to_tensor(obs, device)
        _, next_value = model(obs_tensor)
        next_value = next_value.cpu().numpy()
    
    flat_size = rollout_steps * num_envs
    states = states.reshape(flat_size, 4, 84, 84)
    actions = actions.reshape(flat_size)
    rewards = rewards.reshape(flat_size)
    dones = dones.reshape(flat_size)
    values = values.reshape(flat_size)
    log_probs = log_probs.reshape(flat_size)
    
    returns, advantages = compute_gae(
        rewards, values, dones, 
        np.append(values[1:], next_value),
        gamma=0.99, lam=0.95
    )
    
    return {
        'states': states,
        'actions': actions,
        'rewards': rewards,
        'dones': dones,
        'values': values,
        'log_probs': log_probs,
        'returns': returns,
        'advantages': advantages
    }


def evaluate_policy(model, n_episodes=10, device='cuda'):
    env = gym.make("ALE/Pong-v5", frameskip=1, full_action_space=False, difficulty=1)
    env = AtariPreprocessing(env, grayscale_obs=True, scale_obs=False, frame_skip=4)
    env = FrameStack(env, 4)
    
    returns = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_return = 0
        
        while not done:
            obs_tensor = to_tensor(np.expand_dims(obs, 0), device)
            
            with torch.no_grad():
                logits, _ = model(obs_tensor)
                action = torch.argmax(logits, dim=1).item()
            
            obs, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            episode_return += reward
        
        returns.append(episode_return)
    
    env.close()
    return np.mean(returns)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=2000000, help='Total steps to train')
    parser.add_argument('--num_envs', type=int, default=16, help='Number of environments to run in parallel')
    parser.add_argument('--rollout_steps', type=int, default=128, help='Steps per rollout per environment')
    parser.add_argument('--batch_size', type=int, default=512, help='Minibatch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--clip_ratio', type=float, default=0.1, help='PPO clip ratio')
    parser.add_argument('--vf_coef', type=float, default=0.5, help='Value function coefficient')
    parser.add_argument('--ent_coef', type=float, default=0.02, help='Entropy coefficient - increased for better exploration')
    parser.add_argument('--update_epochs', type=int, default=4, help='Number of PPO epochs per update')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='Device (cuda/cpu)')
    parser.add_argument('--eval_interval', type=int, default=10, help='Evaluate every N updates')
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    initial_difficulty = 0  
    env_fns = [make_env(args.seed + i, difficulty=initial_difficulty) for i in range(args.num_envs)]
    envs = SyncVectorEnv(env_fns)
    
    device = torch.device(args.device)
    model = ActorCritic(4, 6).to(device)  
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    log_path = "online_ppo_pong_training_log.csv"
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['update', 'total_steps', 'mean_reward', 'eval_return'])
    
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/online_ppo_pong_init.pt")
    
    total_steps = 0
    best_eval_return = -21.0  
    updates = 0
    
    curriculum_threshold = -15.0  
    current_difficulty = initial_difficulty
    max_difficulty = 3  
    difficulty_step = 1  
    
    print(f"Starting training for {args.steps} steps with {args.num_envs} environments")
    print(f"Device: {args.device}, Batch size: {args.batch_size}")
    
    start_time = time.time()
    
    while total_steps < args.steps:
        rollout_data = collect_rollout(envs, model, args.rollout_steps, device)
        
        ppo_update(
            model=model,
            optimizer=optimizer,
            states=rollout_data['states'],
            actions=rollout_data['actions'],
            old_log_probs=rollout_data['log_probs'],
            returns=rollout_data['returns'],
            advantages=rollout_data['advantages'],
            clip_ratio=args.clip_ratio,
            vf_coef=args.vf_coef,
            ent_coef=args.ent_coef,
            update_epochs=args.update_epochs,
            batch_size=args.batch_size,
            device=device
        )
        
        # Update stats
        steps_taken = args.num_envs * args.rollout_steps
        total_steps += steps_taken
        updates += 1
        
        # Calculate mean reward per episode
        mean_reward = np.sum(rollout_data['rewards']) / np.sum(rollout_data['dones'])
        
        # Evaluate policy periodically
        if updates % args.eval_interval == 0:
            eval_return = evaluate_policy(model, n_episodes=5, device=device)
            
            # Log results
            with open(log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([updates, total_steps, mean_reward, eval_return])
            
            # Save model if it's the best so far
            if eval_return > best_eval_return:
                best_eval_return = eval_return
                torch.save(model.state_dict(), "models/online_ppo_pong_best.pt")
                print(f"Update {updates} | Steps {total_steps} | Return {eval_return:.1f} | NEW BEST!")
            else:
                print(f"Update {updates} | Steps {total_steps} | Return {eval_return:.1f} | Best: {best_eval_return:.1f}")
            
            # Apply curriculum learning: increase difficulty if performance is good
            if best_eval_return >= curriculum_threshold and current_difficulty < max_difficulty:
                current_difficulty = min(current_difficulty + difficulty_step, max_difficulty)
                # Create new vectorized environments with updated difficulty
                envs.close()
                env_fns = [make_env(args.seed + i, difficulty=current_difficulty) for i in range(args.num_envs)]
                envs = SyncVectorEnv(env_fns)
                print(f"Curriculum learning: Increasing difficulty to {current_difficulty}")
                # Update threshold for next difficulty increase
                curriculum_threshold += 5.0  # Make it harder to reach next level
            
            # Save intermediate checkpoint
            if updates % (args.eval_interval * 10) == 0:
                torch.save(model.state_dict(), f"models/online_ppo_pong_{updates}.pt")
            
            # Early stopping if we reach a good score
            if eval_return >= 20.0:
                print(f"Reached excellent performance with return {eval_return:.1f}")
                print(f"Early stopping at {total_steps} steps")
                break
    
    # Save final model
    torch.save(model.state_dict(), "models/online_ppo_pong_final.pt")
    
    # Final evaluation
    final_return = evaluate_policy(model, n_episodes=10, device=device)
    print(f"Training completed in {time.time() - start_time:.2f} seconds")
    print(f"Total updates: {updates}, Total steps: {total_steps}")
    print(f"Final evaluation return: {final_return:.1f}")
    
    # Close environments
    envs.close()
    
    # Save time comparison for the report
    with open("training_times.txt", "a") as f:
        f.write(f"Online PPO completed in {time.time() - start_time:.2f} seconds\n")
    
    return 0


if __name__ == "__main__":
    main()
