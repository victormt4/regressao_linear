import matplotlib.pyplot as plt
import pandas as pd
import numpy

dataset = pd.read_csv('data/Consumo_cerveja.csv', sep=';')

print('\nMatriz de correlação:\n')
print(dataset.corr().round(4))
