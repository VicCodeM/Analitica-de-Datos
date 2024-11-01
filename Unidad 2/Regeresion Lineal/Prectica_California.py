import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Cargar datos
california_housing = fetch_california_housing()


df_housing = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
df_housing['MedHouseVal'] = california_housing.target
print(df_housing.head())

# variables independientes
variables_independientes = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']


x = df_housing[variables_independientes].values
y = df_housing['MedHouseVal'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# entrenar el modelo
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predecir
y_pred = regressor.predict(x_test)

# Seleccionar una fila específica 
fila_especifica = df_housing.iloc[0]

# Extraer los valores
valores_caracteristicas = fila_especifica[['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']].values

# Predecir un valor individual
predicted_value = regressor.predict([valores_caracteristicas])
print(f"Valor Predicho de la Vivienda Mediana: {predicted_value[0]}")

# Calcular el score
sc = regressor.score(x_test, y_test)
print(f"Puntuación del Modelo: {sc}")

# Graficar
sns.set_theme()
fig, ax = plt.subplots(1, 2, figsize=(10, 10), sharey=True)

# Gráfica para el conjunto de entrenamiento
ax[0].scatter(x_train[:, 0], y_train, label='MedInc')  
ax[0].scatter(x_train[:, 1], y_train, label='HouseAge') 
ax[0].plot(x_train[:, 0], regressor.predict(x_train), c='g', label='Predicción')
ax[0].set_title('MedInc vs MedHouseVal (Conjunto de Entrenamiento)')
ax[0].set_xlabel('MedInc')
ax[0].set_ylabel('MedHouseVal')
ax[0].legend()

# Gráfica para el conjunto de prueba
ax[1].scatter(x_test[:, 0], y_test, label='MedInc')  
ax[1].scatter(x_test[:, 1], y_test, label='HouseAge')  
ax[1].plot(x_test[:, 0], y_pred, c='g', label='Predicción')
ax[1].set_title('MedInc vs MedHouseVal (Conjunto de Prueba)')
ax[1].set_xlabel('MedInc')
ax[1].set_ylabel('MedHouseVal')
ax[1].legend()

plt.suptitle('Modelo de regrsion final con variables independientes')
plt.show()