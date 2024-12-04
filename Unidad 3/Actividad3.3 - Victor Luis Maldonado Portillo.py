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
