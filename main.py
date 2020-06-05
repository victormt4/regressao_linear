import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# setando estilo dos gráficos
sns.set_palette('Accent')
sns.set_style('darkgrid')

dataset = pd.read_csv('data/Consumo_cerveja.csv', sep=';')

print('\nMatriz de correlação:\n')
print(dataset.corr().round(4))

fig, ax = plt.subplots(figsize=(20, 6))

ax = dataset['consumo'].plot()
ax.set_title('Consumo de Cerveja', fontsize=20)
ax.set_ylabel('Litros', fontsize=16)
ax.set_xlabel('Dias', fontsize=16)

plt.show()

# Investigando variável dependente (y) segundo determinada característica
# 0 = dia útil, 1 = fim de semana
ax_boxplot = sns.boxplot(data=dataset, y='consumo', x='fds', orient='v', width=0.2)
ax_boxplot.figure.set_size_inches(12, 6)
ax_boxplot.set_title('Consumo de Cerveja', fontsize=14)
ax_boxplot.set_ylabel('Litros', fontsize=12)
ax_boxplot.set_xlabel('Final de Semana', fontsize=12)

plt.show()
