"""
1. Downloads RL-Unplugged Pong trace
2. Converts to NumPy arrays
3. Builds a d3rlpy MDPDataset that everything else can read
"""
import argparse, pathlib, numpy as np, tensorflow_datasets as tfds
from d3rlpy.dataset import MDPDataset
from tqdm import tqdm

def preprocess(obs):
    # RL-Unplugged Atari frames are already (84,84,1) uint8 → cast to float32 [0,1]
    return (obs.astype(np.float32) / 255.0).squeeze(-1)

def build_dataset(config="Pong_run_5", limit=None, out="pong_offline.h5"):
    builder = tfds.builder("dqn_replay_atari", config="Pong")
    builder.download_and_prepare()
    ds = builder.as_dataset(split="train", shuffle_files=True)

    obs, acts, rews, next_obs, dones = [], [], [], [], []
    for sample in tqdm(tfds.as_numpy(ds), desc="Converting"):
        obs.append(preprocess(sample["observation"]))
        acts.append(sample["action"])
        rews.append(sample["reward"])
        next_obs.append(preprocess(sample["next_observation"]))
        dones.append(sample["step_type"] == 2)  # 2 = LAST
        if limit and len(obs) >= limit: break

    mdp = MDPDataset(
        observations = np.array(obs,  dtype=np.float32),
        actions      = np.array(acts, dtype=np.int64),
        rewards      = np.array(rews, dtype=np.float32),
        terminals    = np.array(dones, dtype=bool),
        episode_terminals = np.array(dones, dtype=bool),
    )
    mdp.dump(out)
    print(f"Saved {len(mdp)} transitions → {out}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="Pong_run_5")
    p.add_argument("--limit", type=int, default=None,
                   help="debug-mode: keep only N transitions")
    p.add_argument("--out"  , default="pong_offline.h5")
    build_dataset(**vars(p.parse_args()))
