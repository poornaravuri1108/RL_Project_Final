#!/usr/bin/env python3
"""
Plot training results from CSV log files for DQN and PPO agents.
"""

import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_training_results():
    # Define the CSV files and their column names
    dqn_file = "dqn_logs.csv"
    ppo_file = "pong_offline_training_log.csv"
    
    plt.figure(figsize=(12, 6))
    
    # Plot DQN results if available
    if os.path.exists(dqn_file):
        try:
            dqn_data = pd.read_csv(dqn_file)
            if 'step' in dqn_data.columns and 'eval/average_reward' in dqn_data.columns:
                plt.plot(dqn_data['step'], dqn_data['eval/average_reward'], 'b-o', label='DQN')
                print(f"Loaded DQN data: {len(dqn_data)} points")
            else:
                print(f"DQN file has unexpected columns: {dqn_data.columns.tolist()}")
        except Exception as e:
            print(f"Error loading DQN data: {e}")
    else:
        print(f"DQN file '{dqn_file}' not found")
    
    # Plot PPO results if available
    if os.path.exists(ppo_file):
        try:
            ppo_data = pd.read_csv(ppo_file)
            if 'epoch' in ppo_data.columns and 'eval_return' in ppo_data.columns:
                plt.plot(ppo_data['epoch'], ppo_data['eval_return'], 'r-o', label='Offline PPO')
                print(f"Loaded PPO data: {len(ppo_data)} points")
            else:
                print(f"PPO file has unexpected columns: {ppo_data.columns.tolist()}")
        except Exception as e:
            print(f"Error loading PPO data: {e}")
    else:
        print(f"PPO file '{ppo_file}' not found")
    
    # Add labels and legend
    plt.xlabel('Training Steps/Epochs')
    plt.ylabel('Average Reward')
    plt.title('Reinforcement Learning Performance on Pong')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save and show the plot
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=300)
    print("Plot saved as 'training_results.png'")
    plt.show()

if __name__ == "__main__":
    plot_training_results()
