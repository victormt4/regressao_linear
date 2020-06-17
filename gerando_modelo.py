from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd


def get_model(dataset_columns):
    dataset = pd.read_csv('data/Consumo_cerveja.csv', sep=';')

    # Criando uma series com a variável dependente (y)
    y = dataset['consumo']

    # Criando um DataFrame com as variáveis explicativas (X)
    X = dataset[dataset_columns]

    # Criando os datasets de treino e teste
    # test size aloca 0.3% do dataset para teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2811)

    model = LinearRegression()
    model.fit(X=X_train, y=y_train)

    return model, X_train, X_test, y_train, y_test
