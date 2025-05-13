#!/bin/bash
#SBATCH --job-name=ppo_pong
#SBATCH --output=ppo_pong_%j.out
#SBATCH --error=ppo_pong_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=poorna.ravuri@sjsu.edu

#SBATCH --partition=lgpuq
#SBATCH --nodelist=g19
##SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=12:00:00

echo "Job ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"
nvidia-smi

export http_proxy=http://172.16.1.2:3128
export https_proxy=http://172.16.1.2:3128
#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd $SLURM_SUBMIT_DIR
source .venv/bin/activate

echo "Installing dependencies…"
pip install --upgrade pip --no-cache-dir
pip install -r offline_pong/requirements.txt --no-cache-dir

echo "Python: $(which python)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
python --version

cd offline_pong
echo "Starting PPO training at $(date)..."

# Redirect all output to log files
exec > >(tee -a "ppo_training_${SLURM_JOB_ID}.log") 2>&1

srun python train_offline_ppo.py \
      --dataset pong_offline_diverse.h5 \
      --epochs 30 \
      --device cuda

# Run validation and log results
echo "Running validation..."
python validate_model.py --model pong_offline_diverse_best_ppo.pt --type ppo --episodes 10

echo "Finished at $(date)"

