#!/bin/bash
#SBATCH --job-name=run_v_pointnext_2048
#SBATCH -N 1
#SBATCH -o /ibex/scratch/ahmems0a/pointnext/cls/run_v_2048.out
#SBATCH -e /ibex/scratch/ahmems0a/pointnext/cls/run_v_2048.err
#SBATCH --mail-user=mahmoud.ahmed@kaust.edu.sa
#SBATCH --mail-type=FAIL,TIME_LIMIT,TIME_LIMIT_90
#SBATCH --time=20:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100:1

conda activate openpoints

python examples/compat_cls/main.py --cfg cfgs/compat_cls/fine/pointnext-s.yaml npoints=2048 timing=True flops=True --data_name fine
