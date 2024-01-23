#!/bin/bash
#SBATCH -J single
#SBATCH -o single_%j.out
#SBATCH -e single_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=48gb

#-----------------Módulos---------------------------
module load miniconda3
source activate encoder_proteins

# ----------------Comandos--------------------------
python /home/dmedina/dna_binding/src/coding_strategies/coding_using_pretrained_models.py /home/dmedina/dna_binding/dataset_to_encode/single_double_dna.csv /home/dmedina/dna_binding/coding_dataset/single_double/reduced/ type_interaction