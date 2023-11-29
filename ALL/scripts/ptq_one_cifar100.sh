#!/bin/bash
if [ ! -d "ret_one/$1" ]; then
    mkdir -p "ret_one/$1"
fi 

sbatch --job-name=$1 -o "ret_one/%x/%j.out" -e "ret_one/%x/%j.err" --export=Model=$1,Dataset=cifar100 ptq_one.slurm
