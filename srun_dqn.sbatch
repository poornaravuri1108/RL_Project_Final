#!/bin/bash
#SBATCH --job-name=dqn_pong
#SBATCH --output=dqn_pong_%j.out
#SBATCH --error=dqn_pong_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=poorna.ravuri@sjsu.edu

#SBATCH --partition=lgpuq
#SBATCH --nodelist=g16
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

echo "Starting DQN training at $(date)..."

# Redirect all output to log files
exec > >(tee -a "dqn_training_${SLURM_JOB_ID}.log") 2>&1

srun python train_dqn.py \
      --dataset pong_offline_diverse.h5 \
      --steps 500000 \
      --batch-size 128 \
      --use-cql \
      --device cuda

# Run validation and log results
echo "Running validation..."
python validate_model.py --model dqn_pong.pt --type dqn --episodes 10

echo "Finished at $(date)"
