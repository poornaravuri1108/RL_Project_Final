#!/bin/bash
<<<<<<< HEAD
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
source ../.venv/bin/activate

=======
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
>>>>>>> a4d6c138877469aa0bd0e20460fe7d35fdd64e16
echo "Python: $(which python)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
python --version

<<<<<<< HEAD
# go into your offline_pong folder
cd offline_pong

=======
cd $RL_Pong_Project/RL_Project_Final/offline_pong
# Run training
>>>>>>> a4d6c138877469aa0bd0e20460fe7d35fdd64e16
srun python train_dqn.py \
      --dataset pong_offline.h5 \
      --steps 500000 \
      --batch-size 64 \
      --device cuda

<<<<<<< HEAD
echo "Done at $(date)"
=======
echo "Finished at $(date)"
>>>>>>> a4d6c138877469aa0bd0e20460fe7d35fdd64e16
