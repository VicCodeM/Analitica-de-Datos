#validación cruzada..
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore

# Paso 1: Cargar el dataset
data = pd.read_csv('salesproject.csv')

# Verificar si hay valores nulos y duplicados
valores_nulos = data.isnull().sum()
print("Valores nulos por columna:\n", valores_nulos)

datos_duplicados = data.duplicated().sum()
print(f"Número de filas duplicadas en el dataset: {datos_duplicados}")

# Eliminar columnas innecesarias
columnas_a_eliminar = ['Invoice ID', 'City', 'Customer type', 'Date', 'Time', 'Payment', 'cogs', 
                        'gross margin percentage', 'gross income', 'Tax 5%']
data = data.drop(columns=columnas_a_eliminar)

# Verificar la distribución de las características
data[['Unit price', 'Quantity', 'Rating']].hist(bins=30, figsize=(10, 5))
plt.show()

# Eliminar outliers usando z-score
z_scores = np.abs(zscore(data[['Unit price', 'Quantity', 'Rating']]))
data = data[(z_scores < 3).all(axis=1)]

# Introducir ruido controlado
np.random.seed(42)
data['Unit price'] += np.random.normal(0, 2, size=len(data))
data['Quantity'] += np.random.normal(0, 0.5, size=len(data))
data['Rating'] += np.random.normal(0, 0.1, size=len(data))

# Definir características y variable objetivo
features = ['Unit price', 'Quantity', 'Rating']
X = data[features]
y = data['Total']  # Regresamos al objetivo original para simplificar

# Dividir el dataset en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Escalar las características
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Paso 2: Modelo de Red Neuronal
modelo_nn = MLPRegressor(hidden_layer_sizes=(100, 50, 25), max_iter=10000, alpha=0.001, random_state=42)

# Validación cruzada
cv_scores_nn = cross_val_score(modelo_nn, X_train, y_train, cv=5, scoring='r2')
print(f"Red Neuronal - R² promedio en validación cruzada: {cv_scores_nn.mean():.2f}")

# Entrenar el modelo de red neuronal
modelo_nn.fit(X_train, y_train)

# Predicciones y evaluación para la red neuronal
y_pred_nn = modelo_nn.predict(X_test)
rmse_nn = mean_squared_error(y_test, y_pred_nn, squared=False)
r2_nn = r2_score(y_test, y_pred_nn)
print(f"Red Neuronal - RMSE: {rmse_nn:.2f}, R²: {r2_nn:.2f}")

# Paso 3: Modelo de Regresión Lineal
modelo_lr = LinearRegression()
modelo_lr.fit(X_train, y_train)

# Predicciones y evaluación para regresión lineal
y_pred_lr = modelo_lr.predict(X_test)
rmse_lr = mean_squared_error(y_test, y_pred_lr, squared=False)
r2_lr = r2_score(y_test, y_pred_lr)
print(f"Regresión Lineal - RMSE: {rmse_lr:.2f}, R²: {r2_lr:.2f}")

# Paso 4: Modelo de Árbol de Decisión
modelo_dt = DecisionTreeRegressor(random_state=42)
modelo_dt.fit(X_train, y_train)

# Predicciones y evaluación para árbol de decisión
y_pred_dt = modelo_dt.predict(X_test)
rmse_dt = mean_squared_error(y_test, y_pred_dt, squared=False)
r2_dt = r2_score(y_test, y_pred_dt)
print(f"Árbol de Decisión - RMSE: {rmse_dt:.2f}, R²: {r2_dt:.2f}")

# Paso 5: Ejemplo de predicción con datos nuevos
nuevos_datos = pd.DataFrame({
    'Unit price': [80],  # Dentro del rango de entrenamiento
    'Quantity': [5],
    'Rating': [7.0]
})

# Escalar los datos nuevos
nuevos_datos_scaled = scaler.transform(nuevos_datos)

# Predicción con la red neuronal
prediccion_nn = modelo_nn.predict(nuevos_datos_scaled)
print(f"Predicción del total de la venta con red neuronal: {prediccion_nn[0]:.2f}")
