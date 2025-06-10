#! /bin/bash
#SBATCH --mem=125G
#SBATCH --cpus-per-task=8
#SBATCH -t 5-03:31:13
#SBATCH -o ../../out/Grievous_Realignment/%x.%A.out
#SBATCH -e ../../err/Grievous_Realignment/%x.%A.log
#########################################################################
# @author: James V. Talwar
# ABOUT: This script performs clumping for VADEr feature set reduction (as described in paper).
# USAGE: sbatch -p carter-compute --job-name=1000G_Clump_PC Clump_1000G.sh

start_time=$SECONDS # track script run time (optional)

basePath=../../Data/1000G/Merged

plink2 --pfile $basePath/1000G_Signif_Intersected_QCed --clump $basePath/ClumpingSSF.ssf --clump-kb 125 --clump-r2 0.8 --out $basePath/Clumped_1000G_PC

elapsed=$(( SECONDS - start_time ))
eval "echo Total Run Time/Elapsed time for clumping: $(date -ud "@$elapsed" +'$((%s/3600/24)) days %H hr %M min %S sec')"
