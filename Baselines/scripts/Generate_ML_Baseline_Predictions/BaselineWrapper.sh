#!/bin/bash
#SBATCH --mem=10G 
#SBATCH --cpus-per-task=2     
#SBATCH -t 11-03:31:13 
#SBATCH -p carter-compute
#SBATCH --array=1-3%2
#SBATCH -o ../../../out/BaselineClassifiers/PC/Generate_ML_Baseline_Predictions/%x.%a.out    
#SBATCH -e ../../../err/BaselineClassifiers/PC/Generate_ML_Baseline_Predictions/%x.%a.err 
########################################################################################
#Usage: sbatch --job-name=XGB_Preds BaselineWrapper.sh 8

dataset=(val test train)
currentDataset=${dataset[$SLURM_ARRAY_TASK_ID-1]}

snpSet=$1

if [ "$currentDataset" == "train" ]; then
    jobName=XGB_Train_Predictions_5e-$snpSet
elif [ "$currentDataset" == "val" ]; then
    jobName=XGB_Val_Predictions_5e-$snpSet
else
    jobName=XGB_UKBB_Predictions_5e-$snpSet
fi

sbatch --wait --job-name=$jobName -p carter-compute BaselineWinners.sh $currentDataset $snpSet

elapsed=$(( SECONDS - start_time ))
eval "echo Total Script Run Time/Elapsed time: $(date -ud "@$elapsed" +'$((%s/3600/24)) days %H hr %M min %S sec')"



