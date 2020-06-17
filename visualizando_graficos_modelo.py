import seaborn as sns
import matplotlib.pyplot as plt
from gerando_modelo import get_model

model, X_train, X_test, y_train, y_test = get_model(['temp_max', 'chuva', 'fds'])

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
