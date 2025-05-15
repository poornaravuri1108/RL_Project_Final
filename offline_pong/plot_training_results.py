"""
Plot training results from CSV log files for DQN and PPO agents (both online and offline).
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def plot_training_results():
    # Define files to look for
    files = [
        {"file": "dqn_logs.csv", "label": "DQN", "x_col": "step", "y_col": "eval/average_reward", "color": "blue", "marker": "o"},
        {"file": "pong_offline_training_log.csv", "label": "Offline PPO", "x_col": "epoch", "y_col": "eval_return", "color": "red", "marker": "s"},
        {"file": "online_ppo_pong_training_log.csv", "label": "Online PPO", "x_col": "update", "y_col": "eval_return", "color": "green", "marker": "^"}
    ]
    
    # Create figure with two subplots - one for rewards, one for losses/entropy
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=False)
    
    # Plot rewards
    for file_info in files:
        file_path = file_info["file"]
        if os.path.exists(file_path):
            try:
                data = pd.read_csv(file_path)
                x_col = file_info["x_col"]
                y_col = file_info["y_col"]
                
                if x_col in data.columns and y_col in data.columns:
                    # Plot the reward curve
                    ax1.plot(data[x_col], data[y_col], 
                             color=file_info["color"], 
                             marker=file_info["marker"], 
                             label=file_info["label"],
                             alpha=0.8,
                             markersize=5)
                    
                    # Add smoothed line for better visualization
                    if len(data) > 5:
                        window_size = min(5, len(data) // 3)
                        smoothed = data[y_col].rolling(window=window_size, center=True).mean()
                        ax1.plot(data[x_col], smoothed, 
                                 color=file_info["color"], 
                                 linewidth=2,
                                 alpha=1.0)
                    
                    print(f"Loaded {file_info['label']} data: {len(data)} points")
                    
                    # For online PPO, also plot policy loss, value loss, and entropy if available
                    if file_info["label"] == "Online PPO" and all(col in data.columns for col in ["policy_loss", "value_loss", "entropy"]):
                        ax2.plot(data[x_col], data["policy_loss"], 
                                 color="purple", 
                                 label="Policy Loss",
                                 alpha=0.7)
                        ax2.plot(data[x_col], data["value_loss"], 
                                 color="orange", 
                                 label="Value Loss",
                                 alpha=0.7)
                        ax2.plot(data[x_col], data["entropy"], 
                                 color="cyan", 
                                 label="Entropy",
                                 alpha=0.7)
                else:
                    print(f"{file_info['label']} file has unexpected columns: {data.columns.tolist()}")
            except Exception as e:
                print(f"Error loading {file_info['label']} data: {e}")
        else:
            print(f"{file_info['label']} file '{file_path}' not found")
    
    # Configure reward plot
    ax1.set_xlabel('Training Steps/Epochs/Updates')
    ax1.set_ylabel('Average Reward')
    ax1.set_title('RL Agents Performance on Pong')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='lower right')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)  # Zero line
    ax1.axhline(y=20, color='green', linestyle='--', alpha=0.5)  # Perfect score line
    ax1.axhline(y=-21, color='red', linestyle='--', alpha=0.5)  # Worst score line
    
    # Configure loss plot
    ax2.set_xlabel('Updates')
    ax2.set_ylabel('Loss Values / Entropy')
    ax2.set_title('Training Metrics for Online PPO')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    
    # Add text annotations
    plt.figtext(0.02, 0.02, 'Note: Dotted lines show perfect score (21), zero score (0), and worst score (-21)', 
                fontsize=9, ha='left')
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=300)
    print("Plot saved as 'training_results.png'")
    plt.show()

def plot_comparison_with_baselines():
    """Plot a comparison of our agents with theoretical baselines."""
    # Define our agents and their best scores
    agents = [
        {"name": "Random Agent", "score": -21.0, "color": "gray"},
        {"name": "DQN", "score": None, "file": "dqn_logs.csv", "col": "eval/average_reward", "color": "blue"},
        {"name": "Offline PPO", "score": None, "file": "pong_offline_training_log.csv", "col": "eval_return", "color": "red"},
        {"name": "Online PPO", "score": None, "file": "online_ppo_pong_training_log.csv", "col": "eval_return", "color": "green"},
        {"name": "Human Expert", "score": 15.0, "color": "purple"},
        {"name": "Perfect Play", "score": 21.0, "color": "gold"}
    ]
    
    # Extract best scores from files
    for agent in agents:
        if "file" in agent and agent["file"] is not None:
            if os.path.exists(agent["file"]):
                try:
                    data = pd.read_csv(agent["file"])
                    if agent["col"] in data.columns:
                        agent["score"] = data[agent["col"]].max()
                        print(f"Best score for {agent['name']}: {agent['score']:.1f}")
                except Exception as e:
                    print(f"Error loading data for {agent['name']}: {e}")
    
    # Create bar chart
    plt.figure(figsize=(10, 6))
    
    # Filter out agents with no score
    valid_agents = [a for a in agents if a["score"] is not None]
    
    names = [a["name"] for a in valid_agents]
    scores = [a["score"] for a in valid_agents]
    colors = [a["color"] for a in valid_agents]
    
    bars = plt.bar(names, scores, color=colors, alpha=0.7)
    
    # Add score labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}',
                ha='center', va='bottom', rotation=0)
    
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.ylim(bottom=-22, top=max(scores) + 3)
    plt.ylabel('Average Score')
    plt.title('Performance Comparison of Different Agents on Pong')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plt.savefig('agent_comparison.png', dpi=300)
    print("Comparison plot saved as 'agent_comparison.png'")
    plt.show()

if __name__ == "__main__":
    plot_training_results()
