#!/bin/bash
#SBATCH --job-name=dqn_pong             # name of the job
#SBATCH --output=slurm-%j.out           # STDOUT → slurm-<jobid>.out
#SBATCH --error=slurm-%j.err            # STDERR → slurm-<jobid>.err
#SBATCH --mail-type=BEGIN,END,FAIL      # send email at start, end, and on failure
#SBATCH --mail-user=poorna.ravuri@sjsu.edu #Mail
#SBATCH --partition=gpu                  # GPU partition (change if yours differs)
#SBATCH --gres=gpu:1                     # request 1 GPU
#SBATCH --ntasks=1                       # single MPI rank
#SBATCH --cpus-per-task=10               # number of CPU cores
#SBATCH --mem=64G                        # total RAM
#SBATCH --time=12:00:00                  # hh:mm:ss runtime
#SBATCH --kill-on-bad-exit=1             # kill job if any step fails

echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "GPU info:"
nvidia-smi

# Load any necessary modules (adjust to your cluster's setup)
#module load cuda/12.6
#module load python/3.9


# Activate your virtualenv
cd $RL_Pong_Project/RL_Project_Final
source .venv/bin/activate

# Print environment
echo "Python: $(which python)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
python --version

cd $RL_Pong_Project/RL_Project_Final/offline_pong
# Run training
srun python train_dqn.py \
      --dataset pong_offline.h5 \
      --steps 500000 \
      --batch-size 64 \
      --device cuda

echo "Finished at $(date)"
