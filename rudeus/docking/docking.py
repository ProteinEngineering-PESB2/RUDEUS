#!/usr/bin/env python3
# -- coding: utf-8 --
# Created on 25 01 2024
# @authors: Gabriel Cabas Mora
# @contact: <gabriel.cabas@umag.cl>
# RUDEUS, a machine learning classification system for DNA-Binding protein identification.
# Released under MIT License

# RUDEUS, a machine learning classification system for DNA-Binding protein identification.
# David Medina-Ortiz 1,2∗, Iván Moya-Barría 1,3, Gabriel Cabas-Mora 1, Nicole Soto-García 1, Roberto Uribe-Paredes 1.
# 1 Departamento de Ingenieria En Computacion, Universidad de Magallanes, Avenida Bulnes 01855, Punta Arenas, Chile.
# 2 Centre for Biotechnology and Bioengineering, CeBiB, Beauchef 851, Santiago, Chile.
# 3 Departamento de Química, Universidad de Magallanes, Av. Pdte. Manuel Bulnes 01855, Punta Arenas, Chile.
# *Corresponding author


"""
Use lightdock to produce a docking for protein-dna interaction.
"""

import os
class Docking:
    """Preprocess a pdb for amber"""
    def __init__(self, protein_pdb_path, dna_pdb_path, n_cores = None):
        self.protein_pdb_path = protein_pdb_path
        self.dna_pdb_path = dna_pdb_path
        self.count_swarm = None
        self.n_cores = n_cores
        if self.n_cores is None:
            self.n_cores = 1

    def execute_lightdock(self, steps):
        current_path = os.path.realpath(__file__)
        os.chdir(os.path.dirname(self.protein_pdb_path))
        command = f"lightdock3_setup.py {os.path.basename(self.protein_pdb_path)} {os.path.basename(self.dna_pdb_path)} -s 10"
        os.system(command)
        swarm_folders = [file for file in os.listdir() if "swarm" in file]
        self.count_swarm = len(swarm_folders)
        command = f"lightdock3.py setup.json {steps} -s dna -c {self.n_cores}"
        os.system(command)
        for folder in swarm_folders:
            if len(os.listdir()) != 0:
                command = f'echo "cd {folder}; lgd_generate_conformations.py ../{os.path.basename(self.protein_pdb_path)} ../{os.path.basename(self.dna_pdb_path)}  gso_{steps}.out 200 > /dev/null 2> /dev/null;" >> generate_lightdock.list;'
                os.system(command)
                command = f'echo "cd {folder}; lgd_cluster_bsas.py gso_{steps}.out > /dev/null 2> /dev/null;" >> cluster_lightdock.list;'
                os.system(command)
        command = f"ant_thony.py -c {self.n_cores} generate_lightdock.list;"
        os.system(command)
        command = f"ant_thony.py -c {self.n_cores} cluster_lightdock.list;"
        os.system(command)
        command = f"lgd_rank.py {self.count_swarm} {steps};"
        os.system(command)
        command = "lgd_copy_structures.py rank_by_scoring.list;"
        os.system(command)
        command = "rm -r init swarm* cluster_lightdock.list generate_lightdock.list lightdock* setup.json"
        os.system(command)
        os.chdir(os.path.dirname(current_path))
        print("READY")