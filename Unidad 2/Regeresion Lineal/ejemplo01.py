import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Cargar datos
df_diamonds = sns.load_dataset('diamonds')

# Seleccionar variables independientes
variables_independientes = ['x', 'y','z']

# Preparar datos
x = df_diamonds[variables_independientes].values
y = df_diamonds['price'].values

# Dividir datos en entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# Crear y entrenar el modelo
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predecir
y_pred = regressor.predict(x_test)

# Predecir un valor individual
psc = regressor.predict([[0.21, 0.21, 0.21]])  #cambiar aqui
print(psc)

# Calcular el score
sc = regressor.score(x_test, y_test)
print(sc)

# Graficar
sns.set_theme()
fig, ax = plt.subplots(1, 2, figsize=(10, 10), sharey=True)

# Gráfica para el conjunto de entrenamiento
ax[0].scatter(x_train[:, 0], y_train, label='x')  # Usar solo la primera variable 'x' para la gráfica
ax[0].scatter(x_train[:, 1], y_train, label='y')  # Usar solo la segunda variable 'y' para la gráfica
ax[0].plot(x_train[:, 0], regressor.predict(x_train), c='g', label='Predicted')
ax[0].set_title('Carat vs Price (Training Set)')
ax[0].set_xlabel('Carat')
ax[0].set_ylabel('Price, USD')
ax[0].legend()

# Gráfica para el conjunto de prueba
ax[1].scatter(x_test[:, 0], y_test, label='x')  # Usar solo la primera variable 'x' para la gráfica
ax[1].scatter(x_test[:, 1], y_test, label='y')  # Usar solo la segunda variable 'y' para la gráfica
ax[1].plot(x_test[:, 0], y_pred, c='g', label='Predicted')
ax[1].set_title('Carat vs Price (Test Set)')
ax[1].set_xlabel('Carat')
ax[1].set_ylabel('Price, USD')
ax[1].legend()

plt.suptitle('Linear Regression Model with Multiple Independent Variables')
plt.show()