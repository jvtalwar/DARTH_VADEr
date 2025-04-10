#!/bin/bash
#SBATCH --mem=175G 
#SBATCH --cpus-per-task=8     
#SBATCH -t 13-03:31:13 
#SBATCH -o ../../../out/BaselineClassifiers/PC/5e-4/Baseline.%x.%A.%a.out # STDOUT
#SBATCH -e ../../../err/BaselineClassifiers/PC/5e-4/Baseline.%x.%A.%a.err # STDERR
#SBATCH --array=1-11%1
#####################################################################################
#Usage: sbatch -p carter-compute --job-name=Search_Linear_SVM_5e-4 BaselineHyperoptLinearSVM.sh

source activate ~/miniconda3/envs/tf_gpu

start_time=$SECONDS

regularization=(-5 -4 -3 -2 -1 0 1 2 3 4 5) #Example regualarization search space
farfetchd=${regularization[$SLURM_ARRAY_TASK_ID - 1]}

pVal=4

python ../../src/ML_Baseline_Hyperopt.py \
 --feather_path ../../../Data/ellipse/table/ellipse-ukb.5e-0$pVal.zscored.feather \
 --pheno_path ../../../Data/PhenosAndIDs/PC/ellipse.pheno.tsv \
 --train_ids ../../../Data/PhenosAndIDs/PC/ellipse_train_ids.txt \
 --val_ids ../../../Data/PhenosAndIDs/PC/ellipse_val_ids.txt \
 --snp_set ../../../Data/snps/extract/PC/ellipse-ukb.5e-0$pVal.extract.txt \
 --valid_maf_snps ../../../Data/snps/extract/PC/QCed_SNP_Sets/LD_Clumped_r2_0.8.txt \
 --model_type svm_linear \
 --rdb_path ../../../Studies/BaselineClassifiers/PC/5e-$pVal \
 --model_checkpoint_path ../../../Models/PC/BaselineClassifiers/5e-$pVal \
 --regularization $farfetchd \
 --age_stats ../../../Data/TrainingSetStatistics/ELLIPSE/TrainingSetAgeStats.pkl


elapsed=$(( SECONDS - start_time ))
eval "echo Total Script Run Time/Elapsed time: $(date -ud "@$elapsed" +'$((%s/3600/24)) days %H hr %M min %S sec')"

