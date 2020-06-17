from sklearn import metrics
import pandas as pd
import numpy as np
from gerando_modelo import get_model

model, X_train, X_test, y_train, y_test = get_model(['temp_max', 'chuva', 'fds'])
model_2, X2_train, X2_test, y2_train, y2_test = get_model(['temp_media', 'chuva', 'fds'])

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

# Visualizando os coeficientes da regressão
# Intercepto: Mantendo as variáveis explicativas = 0, o efeito médio no consumo de cerveja seria 5951 litros
# X1, X2, X3: Mantendo os outros coeficientes constantes, o acréscimo de 1 unidade em Xi gera uma variação média
# no consumo de cerveja de X1 = 684 litros, X2 = -60 litros, X3 = 5401 litros
index = ['Intercepto', 'Temperatura Máxima', 'Chuva (mm)', 'Final de Semana']
coeficientes = pd.DataFrame(data=np.append(model.intercept_, model.coef_), index=index, columns=['Parâmetros'])
print(coeficientes)

# Gerando previsão pontual
X_input = X_test[0:1]
print(f'Consumo previsto: {model.predict(X_input)[0].round(2)} litros')

# Simulando dados
temp_max = 40
chuva = 0
fds = 1
X_input = [[temp_max, chuva, fds]]
print(f'Consumo previsto simulação: {model.predict(X_input)[0].round(2)} litros\n')