#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=a30:4  
#SBATCH --cpus-per-task=64
#SBATCH --mem=250G    
#SBATCH -t 15-03:31:13
#SBATCH -o ../../out/PC/5e-4/%x.%A.out 
#SBATCH -e ../../err/PC/5e-4/%x.%A.err 
#####################################################################
# @author: James V. Talwar  
# USAGE Example: sbatch -p carter-gpu -A carter-gpu --job-name=Hyperparam##_DARTH-VADEr RiseAsMyApprentice.sh 

source activate ~/miniconda3/envs/Intersect2.0 #Activate environment 

start_time=$SECONDS
nvidia-smi

torchrun --standalone --nproc_per_node=gpu ../src/MakeItTrain.py --config_path ../Configs/VADEr_Best.yaml

elapsed=$(( SECONDS - start_time ))
eval "echo Total Script Run Time/Elapsed time: $(date -ud "@$elapsed" +'$((%s/3600/24)) days %H hr %M min %S sec')"