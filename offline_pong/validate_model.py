import argparse
import numpy as np
import torch
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStack
from tqdm import tqdm

from train_dqn import DQNNetwork
from train_offline_ppo import ActorCritic, preprocess_obs


def evaluate_dqn(model_path, device, episodes=10, render=False):
    render_mode = "human" if render else None
    env = gym.make("ALE/Pong-v5", frameskip=1, obs_type="grayscale", render_mode=render_mode)
    env = AtariPreprocessing(env, frame_skip=4, grayscale_obs=True, scale_obs=False, screen_size=84)
    env = FrameStack(env, 4)
    n_actions = env.action_space.n
    
    model = DQNNetwork(4, n_actions).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    total_rewards = []
    with torch.no_grad():
        for episode in tqdm(range(episodes), desc="Evaluating DQN"):
            obs, _ = env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                obs_np = np.array(obs)
                if obs_np.ndim == 3 and obs_np.shape[-1] in [1, 4]:
                    obs_np = np.transpose(obs_np, (2, 0, 1))
                
                obs_tensor = torch.tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
                q_values = model(obs_tensor)
                action = int(q_values.argmax(dim=1).item())
                
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                done = terminated or truncated
            
            total_rewards.append(episode_reward)
    
    env.close()
    return total_rewards


def evaluate_ppo(model_path, device, episodes=10, render=False, model_type='offline', difficulty=1):
    """Evaluate a PPO model on the Pong environment.
    
    Args:
        model_path: Path to the saved model weights
        device: Device to run the model on
        episodes: Number of episodes to evaluate
        render: Whether to render the environment
        model_type: 'offline' or 'online' to determine preprocessing
        difficulty: Difficulty level of the environment (0-3)
    
    Returns:
        List of episode rewards
    """
    render_mode = "human" if render else None
    base = gym.make("ALE/Pong-v5", frameskip=1, full_action_space=False, 
                   render_mode=render_mode, difficulty=difficulty)
    
    # Apply appropriate preprocessing based on model type
    if model_type == 'offline':
        # Offline PPO uses scale_obs=True
        env = FrameStack(AtariPreprocessing(base, grayscale_obs=True, scale_obs=True, frame_skip=1), 4)
    else:
        # Online PPO uses scale_obs=False
        env = FrameStack(AtariPreprocessing(base, grayscale_obs=True, scale_obs=False, frame_skip=4), 4)
    
    n_actions = env.action_space.n
    
    # Load model
    try:
        model = ActorCritic(4, n_actions).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return []
    
    # Evaluate model
    total_rewards = []
    episode_lengths = []
    
    with torch.no_grad():
        for episode in tqdm(range(episodes), desc=f"Evaluating {model_type.capitalize()} PPO"):
            obs, _ = env.reset()
            done = False
            episode_reward = 0
            steps = 0
            
            while not done:
                # Process observation based on model type
                if model_type == 'offline':
                    tensor = preprocess_obs(np.asarray(obs)).unsqueeze(0).to(device)
                else:
                    # Online PPO expects shape [B, C, H, W]
                    obs_np = np.array(obs)
                    if obs_np.shape[-1] in (1, 4):  
                        obs_np = np.transpose(obs_np, (2, 0, 1))
                    tensor = torch.from_numpy(obs_np).unsqueeze(0).float().to(device)
                
                # Get action from model
                logits, _ = model(tensor)
                action = torch.argmax(logits).item()
                
                # Step environment
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                steps += 1
                done = terminated or truncated
            
            total_rewards.append(episode_reward)
            episode_lengths.append(steps)
            print(f"Episode {episode+1}: Reward = {episode_reward}, Length = {steps}")
    
    # Print summary statistics
    mean_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    mean_length = np.mean(episode_lengths)
    
    print(f"\nEvaluation Results for {model_type.capitalize()} PPO:")
    print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Mean episode length: {mean_length:.1f}")
    print(f"Min/Max rewards: {min(total_rewards):.1f}/{max(total_rewards):.1f}")
    
    env.close()
    return total_rewards


def main():
    parser = argparse.ArgumentParser(description='Validate RL models on Pong environment')
    parser.add_argument('--model_type', choices=['dqn', 'offline_ppo', 'online_ppo'], required=True, 
                        help='Type of model to evaluate: dqn, offline_ppo, or online_ppo')
    parser.add_argument('--model_path', required=True, 
                        help='Path to the model weights')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='Device to run evaluation on (cuda/cpu)')
    parser.add_argument('--episodes', type=int, default=10, 
                        help='Number of episodes to evaluate')
    parser.add_argument('--render', action='store_true', 
                        help='Render the environment')
    parser.add_argument('--difficulty', type=int, default=1, choices=[0, 1, 2, 3], 
                        help='Difficulty level for Pong (0-3)')
    parser.add_argument('--deterministic', action='store_true', 
                        help='Use deterministic action selection (default: True)')
    args = parser.parse_args()
    
    print(f"\nEvaluating {args.model_type} model from {args.model_path}")
    print(f"Device: {args.device}, Episodes: {args.episodes}, Difficulty: {args.difficulty}")
    
    if args.model_type == 'dqn':
        rewards = evaluate_dqn(args.model_path, args.device, args.episodes, args.render)
    elif args.model_type == 'offline_ppo':
        rewards = evaluate_ppo(args.model_path, args.device, args.episodes, args.render, 
                              model_type='offline', difficulty=args.difficulty)
    else:  # online_ppo
        rewards = evaluate_ppo(args.model_path, args.device, args.episodes, args.render, 
                              model_type='online', difficulty=args.difficulty)
    
    # Summary statistics are now printed in the evaluation functions
    if rewards:  # Check if we got valid results
        print(f"\nFinal evaluation complete for {args.model_type}.")
        
        # Provide recommendations based on results
        mean_reward = np.mean(rewards)
        if mean_reward < -15:
            print("\nRecommendation: The model is performing poorly. Consider retraining with:")
            print("- Higher entropy coefficient for better exploration")
            print("- Longer training time")
            print("- Different network architecture")
        elif mean_reward < 0:
            print("\nRecommendation: The model is showing some learning but needs improvement:")
            print("- Try adjusting learning rate")
            print("- Increase batch size")
            print("- Add more regularization to prevent overfitting")
        elif mean_reward < 15:
            print("\nRecommendation: The model is performing reasonably well:")
            print("- Fine-tune hyperparameters for better performance")
            print("- Consider longer training to reach optimal performance")
        else:
            print("\nRecommendation: The model is performing excellently!")
            print("- Consider saving this model as your final version")
            print("- You may want to test it against different difficulty levels")
    else:
        print("\nEvaluation failed. Please check the model path and try again.")
        
    # Print individual rewards for debugging if needed
    if rewards:
        print(f"Individual episode rewards: {rewards}")
        
    return mean_reward if rewards else None


if __name__ == "__main__":
    main()
