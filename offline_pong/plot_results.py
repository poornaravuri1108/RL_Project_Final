import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load training logs
dqn_df = pd.read_csv("dqn_logs.csv")   # columns: step, eval/average_reward
ppo_df = pd.read_csv("ppo_logs.csv")   # columns: step, average_reward

sns.set_style("darkgrid")
plt.plot(dqn_df["step"], dqn_df["eval/average_reward"], label="Offline DQN")
plt.plot(ppo_df["step"], ppo_df["average_reward"], label="Offline PPO")
plt.xlabel("Gradient Steps")
plt.ylabel("Average Pong Return")
plt.legend(loc="lower right")
plt.title("Learning Curves – Pong Offline Dataset")
plt.savefig("learning_curve.png", dpi=200)
plt.show()
