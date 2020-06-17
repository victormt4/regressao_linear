from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

dataset = pd.read_csv('data/Consumo_cerveja.csv', sep=';')

# Criando uma series com a variável dependente (y)
y = dataset['consumo']

# Criando um DataFrame com as variáveis explicativas (X)
X = dataset[['temp_max', 'chuva', 'fds']]
X2 = dataset[['temp_media', 'chuva', 'fds']]

# Criando os datasets de treino e teste
# test size aloca 0.3% do dataset para teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2811)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y, test_size=0.3, random_state=2811)

model = LinearRegression()
model.fit(X=X_train, y=y_train)

model_2 = LinearRegression()
model_2.fit(X=X2_train, y=y2_train)

# Obtendo coeficiente de determinação (R²)
# Medida resumida que diz o quanto a linha de regressão se ajusta aos dados. È um valor entre 0 e 1.
# Quanto mais próximo de 1, melhor
print(f'R² Temp. Máxima = {model.score(X_train, y_train).round(2)}')
print(f'R² Temp. Média = {model_2.score(X2_train, y2_train).round(2)}\n')

# Gerando previsões
y_preview = model.predict(X_test)
y2_preview = model_2.predict(X2_test)

# Obtendo R² da previsão
print('R² da previsão:')
print(f'R² Temp. Máxima = {metrics.r2_score(y_test, y_preview).round(2)}')
print(f'R² Temp. Média = {metrics.r2_score(y2_test, y2_preview).round(2)}\n')

# Gerando previsão pontual
X_input = X_test[0:1]
print(f'Consumo previsto: {model.predict(X_input)[0].round(2)} litros')

# Simulando dados
temp_max = 40
chuva = 0
fds = 1
X_input = [[temp_max, chuva, fds]]
print(f'Consumo previsto simulação: {model.predict(X_input)[0].round(2)} litros\n')

# Visualizando os coeficientes da regressão
# Intercepto: Mantendo as variáveis explicativas = 0, o efeito médio no consumo de cerveja seria 5951 litros
# X1, X2, X3: Mantendo os outros coeficientes constantes, o acréscimo de 1 unidade em Xi gera uma variação média
# no consumo de cerveja de X1 = 684 litros, X2 = -60 litros, X3 = 5401 litros
index = ['Intercepto', 'Temperatura Máxima', 'Chuva (mm)', 'Final de Semana']
coeficientes = pd.DataFrame(data=np.append(model.intercept_, model.coef_), index=index, columns=['Parâmetros'])
print(coeficientes)

# Visualizando gráficos do modelo

y_preview_train = model.predict(X_train)
errors = y_train - y_preview_train

ax = sns.scatterplot(x=y_preview_train, y=y_train)
ax.set_title('Previsão x Real')
ax.set_xlabel('Consumo de Cerveja (litros) - Previsão')
ax.set_ylabel('Consumo de Cerveja (litros) - Real')
plt.show()

ax = sns.scatterplot(x=y_preview_train, y=errors, s=150)
ax.set_title('Previsão Resíduos')
ax.set_xlabel('Consumo de Cerveja (litros) - Previsão')
ax.set_ylabel('Resíduos')
plt.show()

# Checando se a variância dos resíduos é constante
ax = sns.scatterplot(x=y_preview_train, y=errors**2, s=150)
ax.set_title('Previsão Resíduos²')
ax.set_xlabel('Consumo de Cerveja (litros) - Previsão')
ax.set_ylabel('Resíduos²')
plt.show()

ax = sns.distplot(errors)
ax.set_title('Distribuição de Frequências dos Resíduos')
ax.set_xlabel('Litros')
plt.show()
