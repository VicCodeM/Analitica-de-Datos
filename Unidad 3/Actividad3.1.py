import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
import numpy as np

# Cargar el dataset de California
california = fetch_california_housing(as_frame=True)
housing = california.frame

# Para este ejmplo tome las primeras 2000 casas
housing_subset = housing.head(2000)

# Mostrar las primeras 2000 filas del dataset
print(housing_subset.head(2000))

# 1. Gráfica de Líneas: Evolución del precio medio de las casas a lo largo de la latitud
housing_sorted = housing_subset.sort_values(by='Latitude')
plt.figure(figsize=(10, 6))
plt.plot(housing_sorted['Latitude'], housing_sorted['MedHouseVal'], marker='o', linestyle='-', color='blue', label='Precio Medio')
plt.title('Evolución del Precio Medio de las Casas a lo largo de la Latitud (Primeras 100 Casas)')
plt.xlabel('Latitud')
plt.ylabel('Precio Medio de las Casas ($1000)')
plt.grid(True)
plt.legend()

# Ajustar los intervalos en el eje X
plt.xticks(np.arange(min(housing_sorted['Latitude']), max(housing_sorted['Latitude']), 0.5))

plt.show()

# 2. Gráfica de Barras: Distribución del ingreso medio por rango
income_bins = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 13,14, float('inf')]
income_ranges = pd.cut(housing_subset['MedInc'], bins=income_bins)
income_counts = income_ranges.value_counts().sort_index()
plt.figure(figsize=(10, 6))

# Calcular los puntos medios de cada intervalo
x = [f"{interval.left}-{interval.right}" for interval in income_counts.index]

plt.bar(x, income_counts.values, color='skyblue')
plt.title('Distribución del Ingreso Medio por Rango (Primeras 100 Casas)')
plt.xlabel('Rango de Ingreso Medio ($10,000)')
plt.ylabel('Número de Observaciones')
plt.grid(True)
plt.xticks(rotation=45)
plt.show()

# Histograma: Distribución de la edad media de las casas
# Histograma: Distribución de la edad media de las casas
plt.figure(figsize=(10, 6))
plt.hist(housing_subset['HouseAge'], bins=np.arange(0, 55, 5), color='green', alpha=0.7, align='left', rwidth=0.8)
plt.title('Distribución de la Edad Media de las Casas (Primeras 100 Casas)')
plt.xlabel('Edad Media de las Casas (años)')
plt.ylabel('Frecuencia')
plt.grid(True)
plt.xticks(np.arange(1, 48, 2))  # Muestra ticks cada 2 años
plt.show()