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
R2 = metrics.r2_score(y_test, y_preview).round(2)
R2_2 = metrics.r2_score(y2_test, y2_preview).round(2)
print('R² da previsão:')
print(f'R² Temp. Máxima = {R2}')
print(f'R² Temp. Média = {R2_2}\n')

# Utilizando outras métricas de comparação
# EQM = Erro quadrático médio
# REQM = Raiz quadrática média

EQM = metrics.mean_squared_error(y_test, y_preview)
REQM = np.sqrt(EQM).round(2)

EQM_2 = metrics.mean_squared_error(y2_test, y2_preview)
REQM_2 = np.sqrt(EQM_2).round(2)

# R² - Quanto mais próximo de 1, melhor (mais adequado ao modelo)
# EQM e REQM, quando menor, melhor (menos erros)
metrics_temp_max = pd.DataFrame([EQM, REQM, R2], ['EQM', 'REQM', 'R²'], columns=['Métricas - Temp Máxima'])
metrics_temp_media = pd.DataFrame([EQM_2, REQM_2, R2_2], ['EQM', 'REQM', 'R²'], columns=['Métricas - Temp Média'])

print(pd.concat([metrics_temp_max, metrics_temp_media], axis=1))
print('\n')

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