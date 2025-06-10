#! /bin/bash
#SBATCH --mem=255G
#SBATCH --cpus-per-task=20
#SBATCH -t 7-03:31:13
#SBATCH -o ../../out/Grievous_Realignment/%x.%A.out
#SBATCH -e ../../err/Grievous_Realignment/%x.%A.log
#########################################################################
# @author: James V. Talwar
# ABOUT: This script generates an effective ELLIPSE training set specific summary statistics file.
# USAGE: sbatch -p carter-compute --job-name=ELLIPSE_Train_SumStats Generate_ELLIPSE_Train_SumStats.sh

start_time=$SECONDS # track script run time (optional)

commonSNPs=../../Data/grievous_harmonized/PC/IntersectingFeatures/IntersectingSNPs.tsv #<-- generated with grievous intersect
covariates=../../Data/ELLIPSE/Phenos_Covars/Train_Covars_For_GLM.tsv #<-- PCs generated with Calc_ELLIPSE_Train_PCA.sh + AGE added as additional covariate
phenos=../../Data/ELLIPSE/Phenos_Covars/Train_Phenos_For_GLM.tsv
genotypes=../../Data/ELLIPSE/GRIEVOUS_Aligned_Train_For_PCA/ELLIPSE_Train_All_Chroms
writePath=../../Data/ELLIPSE/ELLIPSE_Train_SSF/ELLIPSE_Train.sumstats

plink2 --pfile $genotypes --extract $commonSNPs --covar $covariates --covar-variance-standardize --pheno $phenos --out $writePath --glm skip-invalid-pheno firth-fallback hide-covar omit-ref no-x-sex

elapsed=$(( SECONDS - start_time ))
eval "echo Total Run Time/Elapsed time for ELLIPSE PCA: $(date -ud "@$elapsed" +'$((%s/3600/24)) days %H hr %M min %S sec')"