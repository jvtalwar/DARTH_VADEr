#! /bin/bash
#SBATCH --mem=150G
#SBATCH --cpus-per-task=36
#SBATCH -o ../../../../out/BaselineClassifiers/PC/Stat_Gen/Stat_Gen_Baseline.%x.%A.out 
#SBATCH -e ../../../../err/BaselineClassifiers/PC/Stat_Gen/Stat_Gen_Baseline.%x.%A.err
#########################################################################
# USAGE: sbatch -p carter-compute --job-name=ELLIPSE-Train-SSF-SNP-QC_ELLIPSE-Train-Ref runLDPred2.sh

source activate ~/miniconda3/envs/BaRbossa  #<-- activate R env

start_time=$SECONDS


sumStats=../../../../Data/grievous_harmonized/PC/Conti_SSF/GRIEVOUS_Formatted/MergedSSF.ssf 
ldRef=../../../../Data/LDpred2/PC/ELLIPSE_Reference_Panel/ELLIPSE_All
writePath=../../../../Predictions/PC/LDpred2_grid/ELLIPSE_ALL_REF_Conti_SSF_AllPreds
train=../../../../Data/LDpred2/PC/ELLIPSE_Reference_Panel/Train
val=../../../../Data/LDpred2/PC/ELLIPSE_Reference_Panel/Val
test=../../../../Data/LDpred2/PC/ELLIPSE_Reference_Panel/Test
numChroms=21

Rscript LDPred2.R --sumstats $sumStats --ld_ref $ldRef\
 --train $train --val $val --test $test --num_chroms $numChroms --write_path $writePath    
 
 #--classic_p_subset

elapsed=$(( SECONDS - start_time ))
eval "echo Total LDPred-2 script run time/elapsed time: $(date -ud "@$elapsed" +'$((%s/3600/24)) days %H hr %M min %S sec')"


