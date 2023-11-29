#!/bin/bash
if [ ! -d "ckpt_full/cifar100" ]; then
    mkdir -p "ckpt_full/cifar100"
fi

sbatch --job-name=$1 -o "ret_one/%x/%j.out" -e "ret_one/%x/%j.err" --export=Model=$1,Dataset=cifar100 train_one.slurm