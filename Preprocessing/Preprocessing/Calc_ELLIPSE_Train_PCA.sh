#! /bin/bash
#SBATCH --mem=275G
#SBATCH --cpus-per-task=16
#SBATCH -t 5-03:31:13
#SBATCH -o ../../out/Grievous_Realignment/%x.%A.out
#SBATCH -e ../../err/Grievous_Realignment/%x.%A.log
#########################################################################
# @author: James V. Talwar
# ABOUT: This script computes PCA for the ELLIPSE training set (as needed for train set sum stat generation)
# USAGE: sbatch -p carter-compute --job-name=ELLIPSE_Train_PCA Calc_ELLIPSE_Train_PCA.sh

start_time=$SECONDS # track script run time (optional)

baseFile=../../Data/ELLIPSE/GRIEVOUS_Aligned_Train_For_PCA/LD-Pruned_QC_ELLIPSE_Train_All_Chroms
writePath=../../Data/ELLIPSE/PCA/ELLIPSE-Train_PCA

plink2 --pfile $baseFile --pca approx --out $writePath

elapsed=$(( SECONDS - start_time ))
eval "echo Total Run Time/Elapsed time for ELLIPSE PCA: $(date -ud "@$elapsed" +'$((%s/3600/24)) days %H hr %M min %S sec')"

