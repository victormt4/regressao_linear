import pickle
from gerador_modelo import get_model

output_handler = open('data/modelo_consumo_cerveja', 'wb')
model, X_train, X_test, y_train, y_test = get_model(['temp_max', 'chuva', 'fds'])
pickle.dump(model, output_handler)
output_handler.close()