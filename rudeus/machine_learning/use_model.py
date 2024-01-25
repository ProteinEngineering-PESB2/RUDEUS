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

"""Use joblib saved model"""
import joblib
import warnings
warnings.filterwarnings("ignore")
class UseModel:
    """Use joblib saved model"""
    def __init__(self, joblib_path, dataset):
        self.joblib_path = joblib_path
        self.dataset = dataset
    def predict(self):
        model = joblib.load(self.joblib_path)
        prediction = model.predict(self.dataset)
        return prediction