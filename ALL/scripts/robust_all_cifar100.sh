#!/bin/bash
name_list="ResNet_152 ResNet_50 ResNet_18 MobileNetV2 Inception_BN VGG_19 VGG_16 AlexNet_BN AlexNet"

for name in $name_list; do 
    if [ ! -d "ret_one/$name" ]; then
        mkdir -p "ret_one/$name"
    fi 
    if [ "$name" == "ResNet_152" ] ; then
        TimeLen="3-00:00:00"
        QosType="gpu-long"
    else
        TimeLen="1-06:00:00"
        QosType="gpu-normal"
    fi
    adjust="--adjust"
    sbatch --job-name=$name -o "ret_one/%x/%j.out" -e "ret_one/%x/%j.err" -t $TimeLen --qos=$QosType --export=Model=$name,Dataset=cifar100,Adjust=$adjust robust_one.slurm
    adjust=""
    sbatch --job-name=$name -o "ret_one/%x/%j.out" -e "ret_one/%x/%j.err" -t $TimeLen --qos=$QosType --export=Model=$name,Dataset=cifar100,Adjust=$adjust robust_one.slurm
done