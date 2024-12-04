import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Generar datos de ejemplo
np.random.seed(0)  # Para reproducibilidad

# Datos para el diagrama de dispersión
edades = np.random.randint(20, 70, 100)
salarios = np.random.randint(40000, 250000, 100) + edades * 1000
data_dispersión = {
    'Edad': edades,
    'Salario': salarios
}
df_dispersión = pd.DataFrame(data_dispersión)

# Datos para el gráfico de barras
productos = ['Producto A', 'Producto B', 'Producto C', 'Producto D', 'Producto E']
ventas = np.random.randint(100, 300, 5)
data_barras = {
    'Producto': productos,
    'Ventas': ventas
}
df_barras = pd.DataFrame(data_barras)

# Datos para el gráfico de pastel
regiones = ['Región A', 'Región B', 'Región C', 'Región D']
ventas_regionales = np.random.randint(100, 300, 4)
data_pastel = {
    'Región': regiones,
    'Ventas': ventas_regionales
}
df_pastel = pd.DataFrame(data_pastel)

# Diagrama de Dispersión
plt.figure(figsize=(12, 6))
plt.scatter(df_dispersión['Edad'], df_dispersión['Salario'], color='blue', marker='o', alpha=0.5)
plt.title('Diagrama de Dispersión: Edad vs Salario', fontsize=16)
plt.xlabel('Edad', fontsize=14)
plt.ylabel('Salario', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.axhline(y=np.mean(df_dispersión['Salario']), color='red', linestyle='--', label=f'Media: {np.mean(df_dispersión["Salario"]):.2f}')
plt.legend()
plt.show()

# Gráfico de Barras
plt.figure(figsize=(12, 6))
plt.bar(df_barras['Producto'], df_barras['Ventas'], color='green', alpha=0.7)
plt.title('Ventas de Productos', fontsize=16)
plt.xlabel('Producto', fontsize=14)
plt.ylabel('Ventas', fontsize=14)
plt.ylim(0, max(df_barras['Ventas']) + 50)
plt.text(x=-0.5, y=max(df_barras['Ventas']) + 30, s=f'Total Ventas: {sum(df_barras["Ventas"])}', fontsize=12, color='black')
plt.show()

# Gráfico de Pastel
plt.figure(figsize=(12, 6))
plt.pie(df_pastel['Ventas'], labels=df_pastel['Región'], autopct='%1.1f%%', startangle=140, colors=['red', 'yellow', 'blue', 'green'], explode=[0.1, 0, 0, 0])
plt.title('Distribución de Ventas por Región', fontsize=16)
plt.axis('equal')  # Hace que el pastel sea un círculo
plt.show()