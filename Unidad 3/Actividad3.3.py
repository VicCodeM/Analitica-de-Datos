
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

import pandas as pd
import matplotlib.pyplot as plt

# Cargar el archivo CSV
file_path = 'Litros_y_Servicios_2015_2024_Agosto.csv'
data = pd.read_csv(file_path)

# Limpiar nombres de columnas (eliminar espacios extra)
data.columns = data.columns.str.strip()

# Verificar los nombres de las columnas
print("Nombres de las columnas del dataset:")
print(data.columns)

# Inspeccionar las primeras filas del dataset
print("\nPrimeras filas del dataset:")
print(data.head())

# Limpiar datos eliminar valores nulos en columnas clave
data = data.dropna(subset=['Estado', 'Litros de combustible suministrado', 'Servicios realizados'])

# Agrupación por estado y calculo de totales
state_group = data.groupby('Estado').sum(numeric_only=True)

# Ordenar estados por consumo de combustible
state_group = state_group.sort_values(by='Litros de combustible suministrado', ascending=False)

# Seleccionar los 6 estados con mas servicios
top_states = state_group['Servicios realizados'].sort_values(ascending=False).head(6)

# 1. Stackplot Comparación de Litros de Combustible por Estado en el tiempo
top_states_data = data[data['Estado'].isin(top_states.index)]
time_series_data = top_states_data.groupby(['Aniomes', 'Estado'])['Litros de combustible suministrado'].sum().unstack().fillna(0)

plt.figure(figsize=(12, 6))
plt.stackplot(time_series_data.index, time_series_data.values.T, labels=time_series_data.columns, alpha=0.6)
plt.title('Consumo de Combustible por Estado (Stack Plot)')
plt.xlabel('Fecha (YYYYMM)')
plt.ylabel('Litros de Combustible Suministrado')
plt.legend(title='Estados')
plt.tight_layout()
plt.show()

# 2. Scatter Plot Consumo de Combustible por Estado y Servicios Realizados
plt.figure(figsize=(12, 6))

# Definir colores para los estados
colors = ['b', 'r', 'g', 'm', 'c', 'y', 'orange', 'purple']  # Colores distintos para los primeros estados

# Generar scatter plot para los 6 principales estados
for idx, state in enumerate(top_states.index):
    state_data = data[data['Estado'] == state]
    plt.scatter(state_data['Aniomes'], state_data['Litros de combustible suministrado'], 
                label=state, color=colors[idx % len(colors)])  # Asignar color distinto a cada estado

plt.title('Scatter Plot de Consumo de Combustible por Estado')
plt.xlabel('Fecha (YYYYMM)')
plt.ylabel('Litros de Combustible Suministrado')
plt.legend(title='Estados')
plt.tight_layout()
plt.show()

# 3. Gráfico circular: Proporción de servicios realizados por los 8 principales estados
top_10_states = state_group['Servicios realizados'].sort_values(ascending=False).head(8)  # Selecciona los primeros 8 estados
labels = top_states.index  # Los pricipales estados.
sizes = top_states.values  # Los valores de los servicios realizados
explode = [0.1 if i == sizes.argmax() else 0 for i in range(len(sizes))]  # Resaltar el estado con mayor cantidad de servicios

# Definir colores personalizados para cada estado en el gráfico circular
pie_colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0', '#ffb3e6', '#ff6666', '#ffcc00']

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90, colors=pie_colors)
ax1.axis('equal')  
plt.title("Proporción de Servicios Realizados por los 8 Principales Estados")
plt.show()

