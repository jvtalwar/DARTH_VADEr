#!/bin/bash
#SBATCH --mem=185G 
#SBATCH --cpus-per-task=8     
#SBATCH -t 11-03:31:13 
#SBATCH -o ../../../out/BaselineClassifiers/PC/Generate_ML_Baseline_Predictions/5e-8/%x.%A.out 
#SBATCH -e ../../../err/BaselineClassifiers/PC/Generate_ML_Baseline_Predictions/5e-8/%x.%A.err   

source activate ~/miniconda3/envs/tf_gpu

assert() {
    local value="$1"
    
    case "$value" in
        train|val|test)
            ;;
        *)
            echo "Assertion failed: '$value' is not in {train, val, test}"
            exit 1
            ;;
    esac
}


datasetSplit=$1 #{train, val, test}
snpSet=$2
ageStats=../../../Data/TrainingSetStatistics/ELLIPSE/TrainingSetAgeStats.pkl 


assert $datasetSplit

if [ "$datasetSplit" == "train" ]; then
    echo "Generating TRAIN predictions...\n"

    python ./BaselineWinners.py --snp_set ../../../Data/snps/extract/PC/ellipse-ukb.5e-0$snpSet.extract.txt \
    --feather_path ../../../Data/ellipse/table/ellipse-ukb.5e-04.zscored.feather \
    --pheno_path ../../../Data/PhenosAndIDs/PC/ellipse.pheno.tsv \
    --valid_maf_snps ../../../Data/snps/extract/PC/QCed_SNP_Sets/LD_Clumped_r2_0.8.txt \
    --test_ids ../../../Data/PhenosAndIDs/PC/ellipse_train_ids.txt \
    --model_summary_path ../../../Studies/BaselineClassifiers/PC/5e-$snpSet \
    --model_checkpoint_path ../../../Models/PC/BaselineClassifiers/5e-$snpSet \
    --write_path ../../../Predictions/PC/BaselineClassifiers/5e-$snpSet/Train \
    --age_stats $ageStats \
    --subset xgb

elif [ "$datasetSplit" == "val" ]; then
    echo "Generating VAL predictions...\n"

    python ./BaselineWinners.py --snp_set ../../../Data/snps/extract/PC/ellipse-ukb.5e-0$snpSet.extract.txt \
    --feather_path ../../../Data/ellipse/table/ellipse-ukb.5e-04.zscored.feather \
    --pheno_path ../../../Data/PhenosAndIDs/PC/ellipse.pheno.tsv \
    --valid_maf_snps ../../../Data/snps/extract/PC/QCed_SNP_Sets/LD_Clumped_r2_0.8.txt \
    --test_ids ../../../Data/PhenosAndIDs/PC/ellipse_val_ids.txt \
    --model_summary_path ../../../Studies/BaselineClassifiers/PC/5e-$snpSet \
    --model_checkpoint_path ../../../Models/PC/BaselineClassifiers/5e-$snpSet \
    --write_path ../../../Predictions/PC/BaselineClassifiers/5e-$snpSet/Validation \
    --age_stats $ageStats \
    --subset xgb

else 
    echo "Generating TEST predictions...\n"

    python ./BaselineWinners.py --snp_set ../../../Data/snps/extract/PC/ellipse-ukb.5e-0$snpSet.extract.txt \
    --feather_path ../../../Data/ukb/PC_Table/ukbb.prostate.zscored.feather \
    --pheno_path ../../../Data/PhenosAndIDs/PC/ukb.ellipse.pheno.tsv \
    --valid_maf_snps ../../../Data/snps/extract/PC/QCed_SNP_Sets/LD_Clumped_r2_0.8.txt \
    --test_ids ../../../Data/PhenosAndIDs/PC/ukb.test-ids.txt \
    --model_summary_path ../../../Studies/BaselineClassifiers/PC/5e-$snpSet \
    --model_checkpoint_path ../../../Models/PC/BaselineClassifiers/5e-$snpSet \
    --write_path ../../../Predictions/PC/BaselineClassifiers/5e-$snpSet/Test \
    --age_stats $ageStats \
    --subset xgb

fi


