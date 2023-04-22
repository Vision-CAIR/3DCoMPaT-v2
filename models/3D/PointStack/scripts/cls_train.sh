#!/bin/bash
#SBATCH --job-name=cls_train
#SBATCH -N 1
#SBATCH -o slurm_logs/cls_train.out
#SBATCH -e slurm_logs/cls_train.err
#SBATCH --mail-type=ALL
#SBATCH --time=20:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100:1

source activate compat
cd ..

exp_name = $1
data_name = $2
python train.py --cfg_file cfgs/compat/cls.yaml --exp_name $exp_name --val_steps 1 --data_name $data_name
