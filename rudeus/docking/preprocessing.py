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


"""Preprocess a pdb file for amber"""

import os
from Bio.PDB import PDBParser, PDBIO
from docking.amber_types import amber_types
from lightdock.scoring.dna.data.amber import atoms_per_residue
from lightdock.pdbutil.PDBIO import read_atom_line

class Preprocessing:
    """Preprocess a pdb for amber"""
    def __init__(self, protein_pdb_path, dna_pdb_path, output_folder = None):
        self.protein_pdb_path = protein_pdb_path
        self.dna_pdb_path = dna_pdb_path
        self.protein_structure = None
        self.dna_structure = None
        self.amber_keys = amber_types.keys()
        self.conflict_residues = [a.split("-")[0] for a in self.amber_keys]
        self.protein_residues = ["MET", "LYS", "HIS", "TRP", "ASP",
                                "ASN", "PHE", "TRP", "GLN", "GLY",
                                "ALA", "CYS", "GLU", "ILE", "LEU",
                                "PRO", "SER", "VAL", "ARG", "THR"]
        self.translation = {"H5'":"H5'1", "H5''":"H5'2", "H2'":"H2'1", "H2''":"H2'2"}
        self.output_folder = output_folder
        self.create_dirs()

    def create_dirs(self):
        if self.output_folder is None:
            self.output_folder = os.path.dirname(self.protein_pdb_path) + "/output"
        os.makedirs(self.output_folder, exist_ok=True)

    def __import_structure(self, path):
        parser = PDBParser()
        return parser.get_structure(id=os.path.basename(path).replace(".pdb", ""), file=path)
    
    def __protonize(self, pdb_path):
        pdb_without_protons_path = self.output_folder + "/" + os.path.basename(pdb_path).replace('.pdb', '_noh.pdb')
        pdb_with_protons_path = self.output_folder + "/" + os.path.basename(pdb_path).replace('.pdb', '_h.pdb')
        os.system(f"reduce -Trim {pdb_path} > {pdb_without_protons_path}")
        os.system(f"reduce -BUILD {pdb_without_protons_path} > {pdb_with_protons_path}")
        return pdb_with_protons_path

    def __rename_atoms(self, structure):
        for chain in structure.get_chains():
            for res in chain.get_residues():
                if res.get_resname() in self.conflict_residues:
                    for atom in res.get_atoms():
                        res_name = res.get_resname()
                        atom_name = atom.get_name()
                        if f"{res_name}-{atom_name}" in self.amber_keys:
                            atom.name = amber_types[f"{res_name}-{atom_name}"]
                        elif atom_name in amber_types.values():
                            pass
                        else:
                            if res_name in self.protein_residues:
                                res.detach_child(atom.id)
                else:
                    for atom in res.get_atoms():
                        res.detach_child(atom.id)
        return structure

    def __reindex_residues(self, structure):
        for chain in structure.get_chains():
            for index, res in enumerate(chain.get_residues()):
                res.id = (res.id[0], index + 1, res.id[2])
        return structure

    def __format_atom_name(self,atom_name):
        """Format ATOM name with correct padding"""
        if len(atom_name) == 4:
            return atom_name
        else:
            return " %s" % atom_name

    def write_atom_line(self, atom, output):
        """Writes a PDB file format line to output."""
        if atom.__class__.__name__ == "HetAtom":
            atom_type = "HETATM"
        else:
            atom_type = "ATOM  "
        line = "%6s%5d %-4s%-1s%3s%2s%4d%1s   %8.3f%8.3f%8.3f%6.2f%6.2f%12s\n" % (
            atom_type,
            atom.number,
            self.__format_atom_name(atom.name),
            atom.alternative,
            atom.residue_name,
            atom.chain_id,
            atom.residue_number,
            atom.residue_insertion,
            atom.x,
            atom.y,
            atom.z,
            atom.occupancy,
            atom.b_factor,
            atom.element,
        )
        output.write(line)
        
    def __reduce_amber(self, pdb_path):
        output_path = pdb_path.replace(".pdb", "_reduced_amber.pdb")

        with open(pdb_path, "r", encoding="utf-8") as ih:
            with open(output_path, 'w', encoding="utf-8") as oh:
                for line in ih:
                    line = line.rstrip(os.linesep)
                    if line.startswith("ATOM  "):
                        atom = read_atom_line(line)
                        if atom.residue_name not in atoms_per_residue:
                            print(f"Not supported atom: {atom.residue_name}.{atom.name}")
                        else:
                            if atom.name not in atoms_per_residue[atom.residue_name] and atom.is_hydrogen():
                                try:
                                    atom.name = self.translation[atom.name]
                                    self.write_atom_line(atom, oh)
                                except KeyError:
                                    print(f"Atom not found in mapping: {atom.residue_name}.{atom.name}")
                            else:
                                self.write_atom_line(atom, oh)
        return output_path


    def __export_structure(self, structure, path):
        io=PDBIO()
        io.set_structure(structure)
        io.save(path)

    def run(self):
        """Runs preprocessing pipeline"""
        self.protein_pdb_path = self.__protonize(self.protein_pdb_path)
        self.dna_pdb_path = self.__protonize(self.dna_pdb_path)
        self.protein_structure = self.__import_structure(self.protein_pdb_path)
        self.protein_structure = self.__rename_atoms(self.protein_structure)
        self.protein_structure = self.__reindex_residues(self.protein_structure)
        self.__export_structure(self.protein_structure, self.protein_pdb_path.replace(".pdb", "_final.pdb"))
        self.dna_pdb_path = self.__reduce_amber(self.dna_pdb_path)
        self.dna_structure = self.__import_structure(self.dna_pdb_path)
        self.dna_structure = self.__reindex_residues(self.dna_structure)
        self.__export_structure(self.protein_structure, self.dna_pdb_path.replace(".pdb", "_final.pdb"))
        return self.protein_structure, self.dna_pdb_path