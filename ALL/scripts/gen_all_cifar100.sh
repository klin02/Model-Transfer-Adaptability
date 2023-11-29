#!/bin/bash
name_list="ResNet_152 ResNet_50 ResNet_18 MobileNetV2 Inception_BN VGG_19 VGG_16 AlexNet_BN AlexNet"

for name in $name_list; do 
    if [ ! -d "ret_one/$name" ]; then
        mkdir -p "ret_one/$name"
    fi 
    sbatch --job-name=$name -o "ret_one/%x/%j.out" -e "ret_one/%x/%j.err" --export=Model=$name,Dataset=cifar100 gen_one.slurm
done