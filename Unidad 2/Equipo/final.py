import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Paso 1: Cargar el dataset
data = pd.read_csv('salesproject.csv')

# Verificar si hay valores nulos
valores_nulos = data.isnull().sum()
print("Valores nulos por columna:\n", valores_nulos)

# Verificar si hay datos duplicados
datos_duplicados = data.duplicated().sum()
print(f"Número de filas duplicadas en el dataset: {datos_duplicados}")

# Eliminar columnas innecesarias
columnas_a_eliminar = ['Invoice ID', 'City', 'Customer type', 'Date', 'Time', 'Payment', 'cogs', 
                        'gross margin percentage', 'gross income', 'Tax 5%']
data = data.drop(columns=columnas_a_eliminar)

# Paso 2: Calcular las ventas totales por categoría de producto
ventas_por_categoria = data.groupby('Product line')['Total'].sum().sort_values(ascending=False)

# Paso 3: Visualizar las ventas totales por categoría
ventas_por_categoria.plot(kind='bar', figsize=(10, 6))
plt.title('Ventas Totales por Categoría de Producto')
plt.xlabel('Categoría de Producto')
plt.ylabel('Ventas Totales')
plt.xticks(rotation=45)
plt.show()

# Paso 4: Analizar el impacto del género y tipo de cliente (opcional)
ventas_por_categoria_y_genero = data.groupby(['Product line', 'Gender'])['Total'].sum().unstack()

# Gráfico de barras por género
ventas_por_categoria_y_genero.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Ventas Totales por Categoría y Género')
plt.xlabel('Categoría de Producto')
plt.ylabel('Ventas Totales')
plt.xticks(rotation=45)
plt.show()

# Paso 5: Modelo de predicción sin "Tax 5%"
# Selección de características y variable objetivo (sin la columna "Tax 5%")
features = ['Unit price', 'Quantity', 'Rating']  # Eliminamos "Tax 5%" y otras no utilizadas
X = data[features]
y = data['Total']

# Dividir el dataset en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo de regresión lineal
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Hacer predicciones
y_pred = modelo.predict(X_test)

# Calcular el error cuadrático medio (RMSE) para evaluar el modelo
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"Error cuadrático medio (RMSE) del modelo de regresión lineal: {rmse:.2f}")

# Calcular R² (coeficiente de determinación)
r2 = r2_score(y_test, y_pred)
print(f"Coeficiente de determinación R²: {r2:.2f}")

# Ejemplo de predicción con nuevas entradas sin "Tax 5%"
nuevos_datos = pd.DataFrame({
    'Unit price': [1000],
    'Quantity': [3],
    'Rating': [1.9]
})

# Hacer la predicción
prediccion = modelo.predict(nuevos_datos)
print(f"Predicción del total de la venta: {prediccion[0]:.2f}")