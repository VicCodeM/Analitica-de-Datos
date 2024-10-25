from sklearn.neural_network import MLPRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import pandas as pd

# Cargar el dataset
cal_housing = fetch_california_housing()
X, y = cal_housing.data, cal_housing.target

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Estandarizar los datos eliminando la media y escalando para que la varianza sea igual a 1
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Definir el modelo de red neuronal para regresión
mlp = MLPRegressor(hidden_layer_sizes=(256, 128, 64, 32), activation="relu", max_iter=1000, random_state=42)

# Entrenar el modelo
mlp.fit(X_train, y_train)

# Hacer las predicciones con los datos de prueba
y_pred = mlp.predict(X_test)

# Calcular métricas de regresión
mse = metrics.mean_squared_error(y_test, y_pred)
r2 = metrics.r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Mostrar las primeras predicciones y los valores reales
print("Predicciones:", y_pred[:5]) 
print("Valores reales:", y_test[:5])
