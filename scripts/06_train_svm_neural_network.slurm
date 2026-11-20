#!/bin/bash
#SBATCH --job-name=GSE107451_SVM_NN
#SBATCH --account=pi_mukulika_ray_umb_edu
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/svm_nn_%j.out
#SBATCH --error=logs/svm_nn_%j.err

set -euo pipefail

export OPENBLAS_NUM_THREADS=2
export OMP_NUM_THREADS=2

cd /project/pi_mukulika_ray_umb_edu/ML_Project/GSE107451-ML-Classification

"$PYTHON_BIN" -u scripts/06_train_svm_neural_network.py
