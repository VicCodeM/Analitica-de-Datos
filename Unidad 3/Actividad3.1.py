import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

# Cargar el dataset de California
california = fetch_california_housing(as_frame=True)
housing = california.frame

# Mostrar las primeras filas del dataset
print(housing.head())

# 1. Gráfica de Líneas: Evolución del precio medio de las casas a lo largo de la latitud
housing_sorted = housing.sort_values(by='Latitude')
plt.figure(figsize=(10, 6))
plt.plot(housing_sorted['Latitude'], housing_sorted['MedHouseVal'], marker='o', linestyle='-')
plt.title('Evolución del Precio Medio de las Casas a lo largo de la Latitud')
plt.xlabel('Latitud')
plt.ylabel('Precio Medio de las Casas')
plt.grid(True)
plt.show()

# 2. Gráfica de Barras: Comparación del ingreso medio por distrito
income_counts = housing['MedInc'].value_counts().sort_index()
plt.figure(figsize=(10, 6))
plt.bar(income_counts.index, income_counts.values, color='skyblue')
plt.title('Comparación del Ingreso Medio por Distrito')
plt.xlabel('Ingreso Medio')
plt.ylabel('Número de Distritos')
plt.grid(True)
plt.show()

# Histograma: Distribución de la edad media de las casas
plt.figure(figsize=(10, 6))
plt.hist(housing['HouseAge'], bins=30, color='green', alpha=0.7)
plt.title('Distribución de la Edad Media de las Casas')
plt.xlabel('Edad Media de las Casas')
plt.ylabel('Frecuencia')
plt.grid(True)
plt.show()