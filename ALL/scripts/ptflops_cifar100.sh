#!/bin/bash
if [ ! -d "param_flops/cifar100" ]; then
    mkdir -p "param_flops/cifar100"
fi 
sbatch --export=Dataset=cifar100 get_param_flops.slurm