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
        
        # Improved feature extractor with dropout for regularization
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out, 512), nn.ReLU(),
            nn.Dropout(0.1)  # Add dropout to prevent overfitting
        )
        
        # Policy head
        self.pi = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, n_actions)
        )
        
        # Value head
        self.v = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # Initialize weights properly
        self._initialize_weights()
    
    def _initialize_weights(self):
        # Proper weight initialization can significantly improve training
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    module.bias.data.zero_()

    def forward(self, x):
        # Normalize input
        x = x.float() / 255.0
        
        # Extract features
        features = self.fc(self.conv(x))
        
        # Get policy and value outputs
        policy_logits = self.pi(features)
        values = self.v(features).squeeze(-1)
        
        return policy_logits, values


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


# PPO update with improved stability and exploration
def ppo_update(model, optimizer, states, actions, old_log_probs, returns, advantages,
               clip_ratio=0.1, vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5,
               update_epochs=4, batch_size=64, device='cuda', target_kl=0.01):
    states = torch.FloatTensor(states).to(device)
    actions = torch.LongTensor(actions).to(device)
    old_log_probs = torch.FloatTensor(old_log_probs).to(device)
    returns = torch.FloatTensor(returns).to(device)
    advantages = torch.FloatTensor(advantages).to(device)
    
    # Normalize advantages for stability
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # Apply data augmentation to a portion of the states (10%)
    aug_indices = torch.randperm(len(states))[:int(0.1 * len(states))]
    if len(aug_indices) > 0:
        # Simple augmentation: random noise
        noise = torch.randn_like(states[aug_indices]) * 0.01
        states[aug_indices] = torch.clamp(states[aug_indices] + noise, 0, 255)
    
    dataset = TensorDataset(states, actions, old_log_probs, returns, advantages)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Track statistics for early stopping
    policy_losses = []
    value_losses = []
    entropies = []
    kl_divs = []
    
    for epoch in range(update_epochs):
        epoch_policy_loss = 0
        epoch_value_loss = 0
        epoch_entropy = 0
        epoch_kl = 0
        num_batches = 0
        
        for batch in loader:
            b_states, b_actions, b_old_log_probs, b_returns, b_advantages = batch
            
            # Forward pass
            logits, values = model(b_states)
            
            # Compute action distribution
            dist = torch.distributions.Categorical(logits=logits)
            new_log_probs = dist.log_prob(b_actions)
            entropy = dist.entropy().mean()
            
            # Compute KL divergence for early stopping
            kl_div = (torch.exp(b_old_log_probs) * (b_old_log_probs - new_log_probs)).mean().item()
            
            # PPO clipped objective
            ratio = torch.exp(new_log_probs - b_old_log_probs)
            clip_adv = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * b_advantages
            policy_loss = -torch.min(ratio * b_advantages, clip_adv).mean()
            
            # Value loss with clipping for stability
            values_clipped = b_returns + torch.clamp(
                values - b_returns, -clip_ratio, clip_ratio
            )
            v_loss1 = F.mse_loss(values, b_returns)
            v_loss2 = F.mse_loss(values_clipped, b_returns)
            value_loss = torch.max(v_loss1, v_loss2)
            
            # Total loss
            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy
            
            # Optimize
            optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
            loss.backward()
            # Clip gradients to prevent exploding gradients
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            
            # Track statistics
            epoch_policy_loss += policy_loss.item()
            epoch_value_loss += value_loss.item()
            epoch_entropy += entropy.item()
            epoch_kl += kl_div
            num_batches += 1
        
        # Average losses for this epoch
        avg_policy_loss = epoch_policy_loss / num_batches
        avg_value_loss = epoch_value_loss / num_batches
        avg_entropy = epoch_entropy / num_batches
        avg_kl = epoch_kl / num_batches
        
        policy_losses.append(avg_policy_loss)
        value_losses.append(avg_value_loss)
        entropies.append(avg_entropy)
        kl_divs.append(avg_kl)
        
        # Early stopping based on KL divergence
        if avg_kl > 1.5 * target_kl:
            print(f"Early stopping at epoch {epoch+1}/{update_epochs} due to reaching max KL divergence.")
            break
    
    return {
        'policy_loss': np.mean(policy_losses),
        'value_loss': np.mean(value_losses),
        'entropy': np.mean(entropies),
        'kl': np.mean(kl_divs)
    }


class PongRewardShaping(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev_obs = None
        self.ball_position = None
        self.paddle_position = None
        self.consecutive_good_actions = 0
        self.last_ball_direction = None
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_obs = obs
        self.ball_position = None
        self.paddle_position = None
        self.consecutive_good_actions = 0
        self.last_ball_direction = None
        return obs, info
    
    def _extract_ball_and_paddle(self, obs):
        # This is a simplified version - in a real implementation, 
        # you would use computer vision techniques to extract positions
        # For Atari Pong specifically, we're using a very basic approach
        # This assumes grayscale observations (84x84)
        if len(obs.shape) == 3 and obs.shape[0] == 4:  # Frame stack
            frame = obs[3]  # Use the most recent frame
        else:
            frame = obs
            
        # Very basic position extraction - this could be improved
        # with more sophisticated techniques
        return None, None
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        shaped_reward = reward
        
        # Extract ball and paddle positions (simplified)
        ball_pos, paddle_pos = self._extract_ball_and_paddle(obs)
        
        # Basic reward shaping
        if not terminated:
            if reward > 0:  # Agent scored
                shaped_reward = 2.0
                self.consecutive_good_actions = 0
            elif reward < 0:  # Opponent scored
                shaped_reward = -2.0
                self.consecutive_good_actions = 0
            else:  # No immediate reward
                # Small positive reward for staying alive
                shaped_reward += 0.01
                
                # If we can track the ball and paddle, give additional rewards
                if ball_pos is not None and paddle_pos is not None:
                    # Reward for moving paddle towards ball
                    if abs(paddle_pos - ball_pos) < 0.2:  # Paddle is close to ball
                        shaped_reward += 0.05
                        self.consecutive_good_actions += 1
                    else:
                        self.consecutive_good_actions = 0
                    
                    # Bonus for consistent good positioning
                    if self.consecutive_good_actions > 5:
                        shaped_reward += 0.1
        
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


def evaluate_policy(model, n_episodes=10, device='cuda', render=False, deterministic=True, difficulty=1):
    """Evaluate the policy in the environment.
    
    Args:
        model: The policy model to evaluate
        n_episodes: Number of episodes to evaluate
        device: Device to run the model on
        render: Whether to render the environment
        deterministic: Whether to use deterministic action selection
        difficulty: Difficulty level of the environment
    
    Returns:
        Mean episode return across all episodes
    """
    # Create environment with appropriate wrappers
    env = gym.make("ALE/Pong-v5", frameskip=1, full_action_space=False, 
                  difficulty=difficulty, render_mode="rgb_array" if render else None)
    env = AtariPreprocessing(env, grayscale_obs=True, scale_obs=False, frame_skip=4)
    env = FrameStack(env, 4)
    
    # Track statistics
    returns = []
    episode_lengths = []
    
    # Run evaluation episodes
    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_return = 0
        episode_length = 0
        
        while not done:
            # Convert observation to tensor
            obs_tensor = to_tensor(np.expand_dims(obs, 0), device)
            
            # Get action from policy
            with torch.no_grad():
                logits, value = model(obs_tensor)
                
                if deterministic:
                    # Use deterministic action (argmax)
                    action = torch.argmax(logits, dim=1).item()
                else:
                    # Sample from action distribution
                    probs = F.softmax(logits, dim=1)
                    action_dist = torch.distributions.Categorical(probs=probs)
                    action = action_dist.sample().item()
            
            # Step environment
            obs, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            episode_return += reward
            episode_length += 1
            
            # Render if requested
            if render:
                env.render()
        
        # Record episode statistics
        returns.append(episode_return)
        episode_lengths.append(episode_length)
        
        if episode % 5 == 0 or episode == n_episodes - 1:
            print(f"Eval episode {episode+1}/{n_episodes}: Return: {episode_return:.1f}, Length: {episode_length}")
    
    # Close environment
    env.close()
    
    # Return mean episode return
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    print(f"Evaluation over {n_episodes} episodes: Mean return: {mean_return:.1f} ± {std_return:.1f}")
    
    return mean_return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=2000000, help='Total steps to train')
    parser.add_argument('--num_envs', type=int, default=32, help='Number of environments to run in parallel')
    parser.add_argument('--rollout_steps', type=int, default=256, help='Steps per rollout per environment')
    parser.add_argument('--batch_size', type=int, default=1024, help='Minibatch size')
    parser.add_argument('--lr', type=float, default=2.5e-4, help='Learning rate')
    parser.add_argument('--clip_ratio', type=float, default=0.2, help='PPO clip ratio')
    parser.add_argument('--vf_coef', type=float, default=0.5, help='Value function coefficient')
    parser.add_argument('--ent_coef', type=float, default=0.05, help='Entropy coefficient - increased for better exploration')
    parser.add_argument('--update_epochs', type=int, default=8, help='Number of PPO epochs per update')
    parser.add_argument('--target_kl', type=float, default=0.015, help='Target KL divergence for early stopping')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='Device (cuda/cpu)')
    parser.add_argument('--eval_interval', type=int, default=5, help='Evaluate every N updates')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--gae_lambda', type=float, default=0.95, help='GAE lambda parameter')
    parser.add_argument('--max_grad_norm', type=float, default=0.5, help='Maximum gradient norm for clipping')
    parser.add_argument('--lr_decay', action='store_true', help='Use learning rate decay')
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Initialize environment with curriculum learning
    initial_difficulty = 0  
    env_fns = [make_env(args.seed + i, difficulty=initial_difficulty) for i in range(args.num_envs)]
    envs = SyncVectorEnv(env_fns)
    
    # Set up device and model
    device = torch.device(args.device)
    model = ActorCritic(4, 6).to(device)  
    
    # Set up optimizer with a slightly lower initial learning rate
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, eps=1e-5)
    
    # Set up learning rate scheduler if enabled
    scheduler = None
    if args.lr_decay:
        # Cosine annealing scheduler
        total_updates = args.steps // (args.num_envs * args.rollout_steps)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_updates, eta_min=1e-6)
    
    # Set up logging
    log_path = "online_ppo_pong_training_log.csv"
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['update', 'total_steps', 'mean_reward', 'eval_return', 'policy_loss', 'value_loss', 'entropy', 'kl_div'])
    
    # Create models directory and save initial model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/online_ppo_pong_init.pt")
    
    # Initialize training variables
    total_steps = 0
    best_eval_return = -21.0  
    updates = 0
    
    # Curriculum learning parameters
    curriculum_threshold = -15.0  
    current_difficulty = initial_difficulty
    max_difficulty = 3  
    difficulty_step = 1  
    
    print(f"Starting training for {args.steps} steps with {args.num_envs} environments")
    print(f"Device: {args.device}, Batch size: {args.batch_size}, Learning rate: {args.lr}")
    print(f"Entropy coefficient: {args.ent_coef}, PPO clip ratio: {args.clip_ratio}")
    
    start_time = time.time()
    
    # Main training loop
    while total_steps < args.steps:
        # Collect rollout data
        rollout_data = collect_rollout(envs, model, args.rollout_steps, device)
        
        # Update policy using PPO
        update_info = ppo_update(
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
            device=device,
            target_kl=args.target_kl
        )
        
        # Update learning rate if scheduler is enabled
        if scheduler is not None:
            scheduler.step()
        
        # Update stats
        steps_taken = args.num_envs * args.rollout_steps
        total_steps += steps_taken
        updates += 1
        
        # Calculate mean reward per episode
        mean_reward = np.sum(rollout_data['rewards']) / max(np.sum(rollout_data['dones']), 1)
        
        # Evaluate policy periodically
        if updates % args.eval_interval == 0:
            eval_return = evaluate_policy(model, n_episodes=10, device=device)
            
            # Log results
            with open(log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    updates, 
                    total_steps, 
                    mean_reward, 
                    eval_return,
                    update_info['policy_loss'],
                    update_info['value_loss'],
                    update_info['entropy'],
                    update_info['kl']
                ])
            
            # Save model if it's the best so far
            if eval_return > best_eval_return:
                best_eval_return = eval_return
                torch.save(model.state_dict(), "models/online_ppo_pong_best.pt")
                print(f"Update {updates} | Steps {total_steps} | Return {eval_return:.1f} | KL {update_info['kl']:.4f} | NEW BEST!")
            else:
                print(f"Update {updates} | Steps {total_steps} | Return {eval_return:.1f} | KL {update_info['kl']:.4f} | Best: {best_eval_return:.1f}")
            
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
    
    # Final evaluation with more episodes
    final_return = evaluate_policy(model, n_episodes=20, device=device)
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
