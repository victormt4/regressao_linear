import pickle

input_handler = open('data/modelo_consumo_cerveja', 'rb')
model = pickle.load(input_handler)
input_handler.close()

temp_max = 30.5
chuva = 12.2
fds = 0
input_data = [[temp_max, chuva, fds]]

print(f"{model.predict(input_data)[0].round(2)} litros consumidos")
