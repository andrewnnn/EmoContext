#!/bin/bash
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --time=10:00:00
#SBATCH --mem=8GB

# GPUs
#SBATCH --gres=gpu:4                                            # generic resource required (here requires 4 GPUs)
#SBATCH --mem=16GB                                              # specify memory required per node (here set to 16 GB)

# Notifications
#SBATCH −−mail−type=ALL
#SBATCH −−mail−user=andrew.nguyen03@adelaide.edu.au

module load Python/3.6.1-foss-2016b
module load CUDA/9.0.176
module load cuDNN/7.3.1-CUDA-9.0.176

source $FASTDIR/virtualenvs/emocontext/bin/activate

python src/emocontext/baseline.py -config src/emocontext/testBaseline.config > outlogs/output_process.log
deactivate
