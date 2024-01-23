import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import pandas as pd
import sys
from class_models import classification_model
import os

#funcion para crear los set de datos a entrenar
def create_train_test_split(x_data, response, i):
    X_train, X_test, y_train, y_test = train_test_split(x_data, response, random_state=i, test_size=0.3)
    return X_train, X_test, y_train, y_test

doc_config = open(sys.argv[1], 'r')

df_data = pd.read_csv(doc_config.readline().replace("\n", ""))
name_export = doc_config.readline().replace("\n", "")
iteration = int(doc_config.readline().replace("\n", ""))

doc_config.close()

df_positive = df_data[df_data["dna_interaction"] == 1]
df_negative = df_data[df_data["dna_interaction"] == 0]

print("Exploring iteration: ", iteration)
df_data_negative_shuffle = shuffle(df_negative, random_state=iteration)
df_data_negative_to_train = df_data_negative_shuffle[:len(df_positive)]

df_to_train = pd.concat([df_positive, df_data_negative_to_train], axis=0)

response = df_to_train["dna_interaction"]
df_to_train = df_to_train.drop(columns=["dna_interaction"])

X_train, X_test, y_train, y_test = create_train_test_split(df_to_train, response, iteration)

class_instance = classification_model(
    X_train, 
    y_train, 
    X_test, 
    y_test,
    iteration,
    name_export
)
class_instance.make_exploration()
