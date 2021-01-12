#!/bin/bash
#SBATCH --time=0:30:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=50gb
#SBATCH --nodes=1

module load openmpi
module load cuda/10.1.168
module load cudnn/v7.6.4-cuda101
module load python3/3.6.1
module load miniconda3
module load graphviz
source activate /scratch1/jin016/conda_env/dl_env_monai
PYTHONNOUSERSITE=1
cd /scratch1/jin016/tumor_synthesis/driver/
python vis_ablation_test.py

