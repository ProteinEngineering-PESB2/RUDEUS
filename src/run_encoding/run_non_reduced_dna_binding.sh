#!/bin/bash
#SBATCH -J dna_binding
#SBATCH -o dna_binding_%j.out
#SBATCH -e dna_binding_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=128gb

#-----------------Módulos---------------------------
module load miniconda3
source activate encoder_proteins

# ----------------Comandos--------------------------
python /home/dmedina/dna_binding/src/coding_strategies/coding_pretrained_without_reduced.py /home/dmedina/dna_binding/dataset_to_encode/dna_binding_process.csv /home/dmedina/dna_binding/coding_dataset/dna_binding/non_reduced/