#!/bin/bash
#SBATCH --job-name=run_v_pointnext_1024_test
#SBATCH -N 1
#SBATCH -o /ibex/scratch/ahmems0a/pointnext/seg/coarse/run_v_pointnext_1024_test.out
#SBATCH -e /ibex/scratch/ahmems0a/pointnext/seg/coarse/run_v_pointnext_1024_test.err
#SBATCH --mail-user=mahmoud.ahmed@kaust.edu.sa
#SBATCH --mail-type=FAIL,TIME_LIMIT,TIME_LIMIT_90
#SBATCH --time=35:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100:1

conda activate openpoints


python examples/compat/main.py --cfg cfgs/compat/coarse/pointnext-s_c160.yaml npoints=1024 timing=True flops=True --data_name coarse


