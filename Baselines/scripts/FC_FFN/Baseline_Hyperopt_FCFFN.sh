#!/bin/bash
#SBATCH --mem=335G 
#SBATCH --gpus=rtx5000:2   
#SBATCH -t 16-03:31:13
#SBATCH --cpus-per-task=8    
#SBATCH -o ../../../out/BaselineClassifiers/PC/5e-8/ST.%x.%A.out # STDOUT
#SBATCH -e ../../../err/BaselineClassifiers/PC/5e-8/ST.%x.%A.err # STDERR
########################################################################################
# @author: James V. Talwar
# USAGE Example: sbatch -p carter-gpu -A carter-gpu --job-name=FC_FFN_5e-8 Baseline_Hyperopt_FCFFN.sh

source activate ~/miniconda3/envs/tf_gpu

start_time=$SECONDS
nvidia-smi

for i in {0..1} #0 - number of gpus requested per hyperopt call-1
do
    echo Using GPU number $i  
    CUDA_VISIBLE_DEVICES=$i python ../../src/FFNFavorsTheBold.py \
    --config_path ../../configs/5e-8_FC_FFN_ST.yaml \
    --training_objective ST & 
    sleep 8m 59s #prevent feather write issues where loaders can collide; Also ensure no collision on .db creation...; Update as needed for purposes
done

wait

elapsed=$(( SECONDS - start_time ))
eval "echo Total Script Run Time/Elapsed time: $(date -ud "@$elapsed" +'$((%s/3600/24)) days %H hr %M min %S sec')"
