#!/bin/bash
if [ ! -d "param_flops/cifar10" ]; then
    mkdir -p "param_flops/cifar10"
fi 
sbatch --export=Dataset=cifar10 get_param_flops.slurm