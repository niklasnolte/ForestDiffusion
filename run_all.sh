#!/bin/sh
for dataset in iris wine parkinsons climate_model_crashes concrete_compression yacht_hydrodynamics airfoil_self_noise connectionist_bench_sonar ionosphere qsar_biodegradation seeds glass ecoli yeast libras planning_relax blood_transfusion breast_cancer_diagnostic connectionist_bench_vowel concrete_slump wine_quality_red wine_quality_white california bean tictactoe congress car; do
srun --time=1-00:00:00 --gres=gpu:1 --cpus-per-gpu 10 --constraint volta32gb --partition learnlab python script_generation.py --methods disk --add_missing_data True --imputation_method MissForest --datasets ${dataset} --out_path results/disk_${dataset}.txt 2>&1 > results/disk_${dataset}.log &
done
