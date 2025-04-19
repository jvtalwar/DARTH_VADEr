#! /bin/bash
#SBATCH --partition=carter-compute
#SBATCH -t 9-03:31:13
#SBATCH --mem=250G
#SBATCH -o ../../out/PC/PLINK_Processes/ELLIPSE/Feather_Generation/%x.out # STDOUT
#SBATCH -e ../../err/PC/PLINK_Processes/ELLIPSE/Feather_Generation/%x.err # STDERR

#############################
# Name: predataloader.sh
# Description: Predataloader: Generates a composite-level feather file from a list of SNPs (and individuals if provided).

# Example Usage
# sbatch --job-name=ELLIPSE_Composite runPredataloader.sh ../../Data/ellipse/processed_data/genotypes/all/imputed/imputed_pgen ../../Data/snps/extract/PC/ellipse-ukb.5e-04.extract.txt ../../Data/ellipse/table ellipse-ukb.5e-04
#############################

date
echo $SLURM_JOBID
start_time=$SECONDS

# Input arguments
geno=$1  # Plink2 binary file directory
extract=$2  # Extract/Candidate feature set for feather formulation
out_dir=$3  # Directory to write raw files and compiled raw feather
prefix=$4  # File name prefix for raw and feather files
keepPathPrefix=$5 #(Optional) Path to keep file directory (including the keep path prefix)

echo -e "Recoding pgens in $geno to raws by extracting snps in $extract. Outputting to $out_dir/$prefix"

if [ ! -z $5 ]; then
    sbatch --wait --job-name=makeRawFiles.$SLURM_JOB_NAME makeRawFiles.sh $geno $extract $out_dir/$prefix $keepPathPrefix
else
    sbatch --wait --job-name=makeRawFiles.$SLURM_JOB_NAME makeRawFiles.sh $geno $extract $out_dir/$prefix 
fi

elapsedMakeRaw=$(( SECONDS - start_time ))
eval "echo makeRaw.sh run time/elapsed time: $(date -ud "@$elapsedMakeRaw" +'$((%s/3600/24)) days %H hr %M min %S sec')"

echo -e "Activating conda enviroment in ~/miniconda3/envs/tf_gpu" 
source activate ~/miniconda3/envs/tf_gpu  

echo -e "Compiling *.raw files in $out_dir"
python compileRaw.py --directory $out_dir --out_file $out_dir/$prefix.feather 2> ../../err/PC/PLINK_Processes/ELLIPSE/Feather_Generation/compileRaw.$SLURM_JOB_NAME.err 1> ../../out/PC/PLINK_Processes/ELLIPSE/Feather_Generation/compileRaw.$SLURM_JOB_NAME.out

elapsed=$(( SECONDS - start_time ))
eval "echo Total Script run time/elapsed time: $(date -ud "@$elapsed" +'$((%s/3600/24)) days %H hr %M min %S sec')"