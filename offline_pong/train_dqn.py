"""
Offline value-based baseline using d3rlpy’s DQNConfig (Nature CNN).
"""
import argparse, torch, d3rlpy
from d3rlpy.algos import DQNConfig
from d3rlpy.dataset import MDPDataset

def train(dataset_path: str, steps: int, device: str, seed: int = 42):
    d3rlpy.seed(seed)
    dataset = MDPDataset.load(dataset_path)
    env = d3rlpy.envs.GymEnvMaker(
        "ALE/Pong-v5", obs_type="grayscale", frameskip=4, obs_as_image=True)()


    cfg = DQNConfig(
        dueling=True,
        double=True,
        target_update_interval=8000,
        learning_rate=1e-4,
        batch_size=32,
        encoder_factory="nature_cnn",
        gamma=0.99
    )
    dqn = cfg.create(device=device)
    dqn.fit(
        dataset,
        n_steps=steps,
        scorers={"env": d3rlpy.metrics.evaluate_on_environment(env, n_trials=10)}
    )
    dqn.save_model("dqn_pong.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="pong_offline.h5")
    parser.add_argument("--steps",   type=int, default=1_000_000)
    parser.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    train(args.dataset, args.steps, args.device)
