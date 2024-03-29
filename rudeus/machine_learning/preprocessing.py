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



from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
def random_under_sampling_split(X, y, random_state = None):
    """Get a balanced random sample of data and splits it into training and testing datasets."""
    if random_state is not None:
        rus = RandomUnderSampler(random_state = random_state)
    else:
        rus = RandomUnderSampler()
    X, y = rus.fit_resample(X, y)
    return train_test_split(X, y, random_state=1)

def binarize_target(y):
    """Binarize target"""
    encoder = LabelEncoder()
    return encoder.fit_transform(y)
