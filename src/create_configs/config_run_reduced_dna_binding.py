import sys
import os

path_encoder = "/home/dmedina/dna_binding/coding_dataset/dna_binding/reduced/"

prefix_save_results = "/home/dmedina/dna_binding/training_results/dna_binding/ml_classic/"
prefix_save_configs = "/home/dmedina/dna_binding/src/run_training_double/config/"
iterations = [i for i in range(1, 30)]
iterations.append(42)

list_encoder = os.listdir(path_encoder)
for encoder in list_encoder:
    print("Process encoder: ", encoder)
    encoder_name = encoder.split(".")[0]
    for i in iterations:
        name_config = f"{prefix_save_configs}{encoder_name}_{i}.txt"

        doc_config = open(name_config, 'w')

        line1 = f"{path_encoder}{encoder}\n"
        line2 = f"{prefix_save_results}{encoder_name}_exploring_{i}.csv\n"
        line3 = f"{str(i)}\n"

        doc_config.write(line1)
        doc_config.write(line2)
        doc_config.write(line3)

        doc_config.close()