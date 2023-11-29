#!/bin/bash

#- Job parameters

# (TODO)
# Please modify job name

#SBATCH -J ALL              # The job name
#SBATCH -o ret/ret-%j.out        # Write the standard output to file named 'ret-<job_number>.out'
#SBATCH -e ret/ret-%j.err        # Write the standard error to file named 'ret-<job_number>.err'


#- Resources

# (TODO)
# Please modify your requirements

#SBATCH -p nv-gpu                    # Submit to 'nv-gpu' Partitiion
#SBATCH -t 0-01:30:00                # Run for a maximum time of 0 days, 12 hours, 00 mins, 00 secs
#SBATCH --nodes=1                    # Request N nodes
#SBATCH --gres=gpu:1                 # Request M GPU per node
#SBATCH --gres-flags=enforce-binding # CPU-GPU Affinity
#SBATCH --qos=gpu-debug             # Request QOS Type

###
### The system will alloc 8 or 16 cores per gpu by default.
### If you need more or less, use following:
### #SBATCH --cpus-per-task=K            # Request K cores
###
### 
### Without specifying the constraint, any available nodes that meet the requirement will be allocated
### You can specify the characteristics of the compute nodes, and even the names of the compute nodes
###
### #SBATCH --nodelist=gpu-v00           # Request a specific list of hosts 
### #SBATCH --constraint="Volta|RTX8000" # Request GPU Type: Volta(V100 or V100S) or RTX8000
###

# set constraint for RTX8000 to meet my cuda
#SBATCH --constraint="Ampere|RTX8000|T4"

#- Log information

echo "Job start at $(date "+%Y-%m-%d %H:%M:%S")"
echo "Job run at:"
echo "$(hostnamectl)"

#- Load environments
source /tools/module_env.sh
module list                       # list modules loaded

##- Tools
module load cluster-tools/v1.0
module load slurm-tools/v1.0
module load cmake/3.15.7
module load git/2.17.1
module load vim/8.1.2424

##- language
module load python3/3.6.8

##- CUDA
# module load cuda-cudnn/10.2-7.6.5
# module load cuda-cudnn/11.2-8.2.1
module load cuda-cudnn/11.1-8.2.1

##- virtualenv
# source xxxxx/activate

echo $(module list)              # list modules loaded
echo $(which gcc)
echo $(which python)
echo $(which python3)

cluster-quota                    # nas quota

nvidia-smi --format=csv --query-gpu=name,driver_version,power.limit # gpu info

#- Warning! Please not change your CUDA_VISIBLE_DEVICES
#- in `.bashrc`, `env.sh`, or your job script
echo "Use GPU ${CUDA_VISIBLE_DEVICES}"                              # which gpus
#- The CUDA_VISIBLE_DEVICES variable is assigned and specified by SLURM

#- Job step
# [EDIT HERE(TODO)]
name_list="AlexNet AlexNet_BN VGG_16 VGG_19 Inception_BN ResNet_18 ResNet_50 ResNet_152 MobileNetV2"
# name_list="MobileNetV2"
for name in $name_list; do 
	if [ -f "param_flops/$name.txt" ];then
		echo "$name: param_flops exists"
	# elif [ ! -f "ckpt/cifar10_$name.pt" ];then
	# 	echo "$name: ckpt not exists"
	else
		python get_param_flops.py $name > param_flops/$name.txt
	fi
done

#- End
echo "Job end at $(date "+%Y-%m-%d %H:%M:%S")"
