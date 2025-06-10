#! /bin/bash
#SBATCH --mem=100G
#SBATCH --cpus-per-task=16
#SBATCH -o ../../../out/BaselineClassifiers/PC/Stat_Gen/Stat_Gen_Baseline.%x.%A.out 
#SBATCH -e ../../../err/BaselineClassifiers/PC/Stat_Gen/Stat_Gen_Baseline.%x.%A.err
#########################################################################
# @author: James V. Talwar
# USAGE: sbatch -p carter-compute --job-name=PRSice2_PC ImPRSice2.sh

source activate ~/miniconda3/envs/BaRbossa  #<-- activate R env

start_time=$SECONDS

genotypes=~/Data/Stat_Gen/PC/1000G_Reference_Panel/UKBB_Test/chr#
ldGeno=~/Data/1000G/PC_Binary_Realigned/BedFiles/chr#
phenoFile=../../../Data/PhenosAndIDs/PC/PRSice2/ukb.ellipse.pheno.tsv
prsiceTwoPath=~/programs/PRSice_2
sumStats=../../../Data/grievous_harmonized/PC/Conti_SSF/GRIEVOUS_Formatted/PRSice2_Formatted_MergedSSF.ssf
writePath=../../../Predictions/PC/PRSice2/UKBB-Test 

Rscript $prsiceTwoPath/PRSice.R \
--dir $prsiceTwoPath \
--prsice $prsiceTwoPath/PRSice_linux \
--base $sumStats \
--snp GRIEVOUS_ID \
--chr CHR \
--bp POS \
--A1 ALT \
--A2 REF \
--stat BETA \
--pvalue PVAL \
--beta \
--target $genotypes \
--ld $ldGeno \
--binary-target T \
--score sum \
--missing MEAN_IMPUTE \
--keep-ambig \
--thread $SLURM_CPUS_PER_TASK \
--bar-levels 5e-08,5e-07,5e-06,5e-05,5e-04 \
--model add \
--pheno $phenoFile \
--ignore-fid \
--pheno-col PHENOTYPE \
--fastscore \
--all-score \
--quantile 10 \
--out $writePath

elapsed=$(( SECONDS - start_time ))
eval "echo Total PRSice-2 script run time/elapsed time: $(date -ud "@$elapsed" +'$((%s/3600/24)) days %H hr %M min %S sec')"