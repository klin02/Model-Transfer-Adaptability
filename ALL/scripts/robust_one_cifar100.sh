#!/bin/bash
if [ ! -d "ret_one/$1" ]; then
    mkdir -p "ret_one/$1"
fi 

if [ "$1" == "ResNet_152" ] ; then
    TimeLen="3-00:00:00"
    QosType="gpu-long"
else
    TimeLen="1-06:00:00"
    QosType="gpu-normal"
fi

adjust="--adjust"
sbatch --job-name=$1 -o "ret_one/%x/%j.out" -e "ret_one/%x/%j.err" -t $TimeLen --qos=$QosType --export=Model=$1,Dataset=cifar100,Adjust=$adjust robust_one.slurm
adjust=""
sbatch --job-name=$1 -o "ret_one/%x/%j.out" -e "ret_one/%x/%j.err" -t $TimeLen --qos=$QosType --export=Model=$1,Dataset=cifar100,Adjust=$adjust robust_one.slurm