#! /bin/bash
#SBATCH --partition=carter-compute
#SBATCH --mem=25G
#SBATCH -o ../../out/PC/PLINK_Processes/ELLIPSE/Feather_Generation/%x.chr%a.out # STDOUT
#SBATCH -e ../../err/PC/PLINK_Processes/ELLIPSE/Feather_Generation/%x.chr%a.err # STDERR
#SBATCH --array=1-23%23
# ############################
# Usage: sbatch --job-name=makeRawFiles makeRawFiles.sh $geno_dir $extract $out_path+file_prefix $ids(optional)
# Description: used to extract snps from imputed snp data as input into compile-freq.py
# Date: 09/06/2022
# ############################

date
echo -e "Job ID: $SLURM_ARRAY_JOB_ID"

chroms=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 X)
chrom=${chroms[$SLURM_ARRAY_TASK_ID-1]}

alignedStatus=grievousRealigned.grievousReindexed

# Arguments
geno=$1
extract=$2
prefix=$3 #<-- This is out dir + file name prefix: $out_dir/$prefix


if [ ! -z $4 ]; then

    if [[ "$chrom" == X ]]; then
    
    keep=$4.x.txt
    # Run the commmand
    CMD="plink2 --pfile ${geno}/chr$chrom.$alignedStatus --extract $extract --keep $keep --export A --out $prefix.chr$chrom.feature"
    echo -e "Running $CMD"
    $CMD
    
    else
    
    keep=$4.txt
    # Run the commmand
    CMD="plink2 --pfile ${geno}/chr$chrom.$alignedStatus--extract $extract --keep $keep --export A --out $prefix.chr$chrom.feature"
    echo -e "Running $CMD"
    $CMD
    
    fi
    
else
    echo -e "No provided IDs, using all"
    # Run the commmand
    CMD="plink2 --pfile ${geno}/chr$chrom.$alignedStatus --extract $extract --export A --out $prefix.chr$chrom.feature"
    echo -e "Running $CMD"
    $CMD
fi

date
