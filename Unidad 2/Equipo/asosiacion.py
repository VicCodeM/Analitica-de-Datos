import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import association_rules, apriori
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el conjunto de datos
df = pd.read_csv('salesproject.csv')

# Verificar las primeras filas del DataFrame
print("Primeras filas del DataFrame:")
print(df.head())

# Preprocesar los datos para el análisis de asociación
# Crear una tabla dinámica para contar los productos por transacción
df_pivot = df.pivot_table(index='Invoice ID', columns='Product line', values='Quantity', aggfunc='sum').fillna(0)

# Verificar la tabla dinámica
print("\nTabla dinámica:")
print(df_pivot.head())

# Convertir a tipo entero
df_pivot = df_pivot.astype(int)

# Función para codificar valores: 1 si hay compra, 0 si no
def encode(x):
    if x <= 0:
        return 0
    else:
        return 1

# Aplicar la función de codificación a la tabla dinámica
df_pivot = df_pivot.apply(lambda x: x.map(encode))

# Verificar la tabla dinámica después de la codificación
print("\nTabla dinámica después de la codificación:")
print(df_pivot.head())

# Establecer el soporte mínimo para encontrar ítems frecuentes
support = 0.01

# Encontrar ítems frecuentes usando el algoritmo Apriori
frequent_items = apriori(df_pivot, min_support=support, use_colnames=True)

# Verificar los ítems frecuentes
print("\nÍtems frecuentes:")
print(frequent_items.head())

# Ordenar los ítems frecuentes por soporte
frequent_items.sort_values('support', ascending=True)

# Definir la métrica y el umbral mínimo para las reglas
metric = 'lift'
min_treshold = 3

# Generar reglas de asociación
rules = association_rules(frequent_items, metric=metric, min_threshold=min_treshold)

# Verificar las reglas de asociación
print("\nReglas de asociación:")
print(rules.head())

# Reiniciar los índices y ordenar las reglas por confianza
rules.reset_index(drop=True).sort_values('confidence', ascending=False, inplace=True)

# Mostrar las reglas de asociación
print("\nReglas de asociación ordenadas por confianza:")
print(rules)

# Gráfica de los ítems frecuentes
plt.figure(figsize=(10, 6))
sns.barplot(x='support', y='itemsets', data=frequent_items.sort_values('support', ascending=False).head(10))
plt.title('Top 10 Ítems Frecuentes por Soporte')
plt.xlabel('Soporte')
plt.ylabel('Ítems')
plt.show()

