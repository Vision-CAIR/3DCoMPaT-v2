#!/bin/bash
#SBATCH --job-name=cls_test
#SBATCH -N 1
#SBATCH -o slurm_logs/cls_test.out
#SBATCH -e slurm_logs/cls_test.err
#SBATCH --mail-type=ALL
#SBATCH --time=30:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100:1

source activate compat
cd ..

exp_name = $1
python test.py --cfg_file cfgs/compat/fine.yaml --ckpt experiments/CompatLoader3D/$exp_name/ckpt/ckpt-best.pth --data_name fine
