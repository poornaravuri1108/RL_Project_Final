#!/bin/bash
#SBATCH --job-name=dqn_pong             # your job name
#SBATCH --output=slurm-%j.out           # STDOUT → slurm-<jobid>.out
#SBATCH --error=slurm-%j.err            # STDERR → slurm-<jobid>.err
#SBATCH --mail-type=BEGIN,END,FAIL      # email notifications
#SBATCH --mail-user=poorna.ravuri@sjsu.edu
#SBATCH --partition=mgpuq               # medium-GPU queue
#SBATCH --ntasks=1                      # one process
#SBATCH --cpus-per-task=10              # CPU cores
#SBATCH --mem=64G                       # RAM
#SBATCH --time=12:00:00                 # walltime hh:mm:ss

echo "Job ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"
nvidia-smi
export http_proxy=http://172.16.1.2:3128
export https_proxy=http://172.16.1.2:3128

# Go to the directory where you ran sbatch
cd $SLURM_SUBMIT_DIR

# Activate your virtualenv (located at repo root/.venv)
source .venv/bin/activate

echo "Installing dependencies into .venv…"
pip install --upgrade pip --no-cache-dir
pip install -r offline_pong/requirements.txt --no-cache-dir

echo "Python: $(which python)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
python --version

# Move into your code directory
cd offline_pong

echo "Starting training at $(date)…"
srun python train_dqn.py \
      --dataset pong_offline.h5 \
      --steps 500000 \
      --batch-size 64 \
      --device cuda

echo "Finished at $(date)"
