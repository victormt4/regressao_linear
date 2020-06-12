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

# Distribuição de frequências da variável dependente (y)
ax_distplot = sns.distplot(dataset['consumo'])
ax_distplot.figure.set_size_inches(12, 6)
ax_distplot.set_title('Distribuição de Frequências', fontsize=14)
ax_distplot.set_ylabel('Consumo de Cerveja (Litros)', fontsize=12)

plt.show()

# Visualizando a correlação entre as variáveis
ax_pairplot = sns.pairplot(dataset, y_vars='consumo', x_vars=['temp_min', 'temp_media', 'temp_max', 'chuva', 'fds'])
ax_pairplot.fig.suptitle('Dispersão entre as Variáveis', y=1.05)
plt.show()

# o atributo kind='reg' traça uma reta de regressão
ax_pairplot = sns.pairplot(dataset, y_vars='consumo', x_vars=['temp_min', 'temp_media', 'temp_max', 'chuva', 'fds'],
                           kind='reg')
ax_pairplot.fig.suptitle('Dispersão entre as Variáveis', y=1.05)
plt.show()

ax_joinplot = sns.jointplot(x='temp_max', y='consumo', data=dataset, kind='reg')
ax_joinplot.fig.suptitle('Dispersão - Consumo x Temperatura')
ax_joinplot.set_axis_labels('Temperatura Máxima', 'Consumo de Cerveja')
plt.show()

ax_lmplot = sns.lmplot(x='temp_max', y='consumo', data=dataset, hue='fds', markers=['o', '*'], legend=False)
ax_lmplot.fig.suptitle('Reta de Regressão - Consumo x Temperatura x Final de Semana')
ax_lmplot.set_axis_labels('Temperatura Máxima (ºC)', 'Consumo de Cerveja (litros)')
ax_lmplot.add_legend(title='Fim de semana')
plt.show()
