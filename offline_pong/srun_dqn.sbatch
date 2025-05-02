#!/bin/bash
#SBATCH --job-name=dqn_pong             # your job name
#SBATCH --output=slurm-%j.out           # STDOUT → slurm-<jobid>.out
#SBATCH --error=slurm-%j.err            # STDERR → slurm-<jobid>.err
#SBATCH --mail-type=BEGIN,END,FAIL      # email notifications
#SBATCH --mail-user=poorna.ravuri@sjsu.edu
#SBATCH --partition=mgpuq               # medium‐GPU queue (7 days max)
#SBATCH --ntasks=1                       # one task
#SBATCH --cpus-per-task=10               # CPU cores
#SBATCH --mem=64G                        # RAM
#SBATCH --time=12:00:00                  # hh:mm:ss walltime

echo "Job ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"
nvidia-smi

# (If your cluster needs modules...)
# module load cuda/12.6
# module load python/3.11.7

cd $SLURM_SUBMIT_DIR
source .venv/bin/activate

echo "Python: $(which python)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
python --version

# go into your offline_pong folder
cd offline_pong

srun python train_dqn.py \
      --dataset pong_offline.h5 \
      --steps 500000 \
      --batch-size 64 \
      --device cuda

echo "Done at $(date)"
