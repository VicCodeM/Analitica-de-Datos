import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Paso 1: Cargar el dataset
data = pd.read_csv('salesproject.csv')  # Reemplaza 'path_to_your_dataset.csv' con la ruta de tu archivo

# Mostrar las primeras filas para verificar el dataset
print(data.head())

# Paso 2: Calcular las ventas totales por categoría de producto
ventas_por_categoria = data.groupby('Product line')['Total'].sum().sort_values(ascending=False)

# Mostrar las categorías con más ventas
print("Ventas totales por categoría:")
print(ventas_por_categoria)

# Paso 3: Visualizar las ventas totales por categoría
ventas_por_categoria.plot(kind='bar', figsize=(10, 6))
plt.title('Ventas Totales por Categoría de Producto')
plt.xlabel('Categoría de Producto')
plt.ylabel('Ventas Totales')
plt.xticks(rotation=45)
plt.show()

# Paso 4: Analizar el impacto del género y tipo de cliente (opcional)
# Agrupar por 'Product line' y 'Gender' para ver las ventas por género en cada categoría
ventas_por_categoria_y_genero = data.groupby(['Product line', 'Gender'])['Total'].sum().unstack()

# Mostrar el resultado
print("Ventas por categoría y género:")
print(ventas_por_categoria_y_genero)

# Gráfico de barras por género
ventas_por_categoria_y_genero.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Ventas Totales por Categoría y Género')
plt.xlabel('Categoría de Producto')
plt.ylabel('Ventas Totales')
plt.xticks(rotation=45)
plt.show()

# Paso 5: Modelo de predicción (opcional)
# Ingeniería de características
data['Total Price'] = data['Unit price'] * data['Quantity']
data['Total Price with Tax'] = data['Total Price'] * (1 + data['Tax 5%'])

# Selección de características y variable objetivo
features = ['Unit price', 'Quantity', 'Tax 5%', 'Rating', 'Total Price', 'Total Price with Tax']
categorical_features = ['Product line', 'Gender', 'Customer type', 'Payment']

X = data[features + categorical_features]
y = data['Total']

# Preprocesamiento
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Crear y entrenar el modelo de regresión lineal con regularización
modelo = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', Ridge(alpha=1.0))
])

# Validación cruzada
cv_scores = cross_val_score(modelo, X, y, cv=5, scoring='r2')
print(f"R² promedio con validación cruzada: {cv_scores.mean():.2f}")

# Dividir el dataset en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo
modelo.fit(X_train, y_train)

# Hacer predicciones
y_pred = modelo.predict(X_test)

# Calcular el error cuadrático medio (RMSE) para evaluar el modelo
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"Error cuadrático medio (RMSE) del modelo de regresión lineal: {rmse:.2f}")

# Calcular R² (coeficiente de determinación)
r2 = r2_score(y_test, y_pred)
print(f"Coeficiente de determinación R²: {r2:.2f}")

# Calcular el porcentaje de varianza explicada
porcentaje_varianza_explicada = r2 * 100
print(f"Porcentaje de varianza explicada: {porcentaje_varianza_explicada:.2f}%")

# Ejemplo de predicción con nuevas entradas
nuevos_datos = pd.DataFrame({
    'Unit price': [1000],
    'Quantity': [3],
    'Tax 5%': [7.5],
    'Rating': [1.9],
    'Total Price': [3000],
    'Total Price with Tax': [3075],
    'Product line': ['Health and beauty'],
    'Gender': ['Female'],
    'Customer type': ['Member'],
    'Payment': ['Ewallet']
})

# Hacer la predicción
prediccion = modelo.predict(nuevos_datos)
print(f"Predicción del total de la venta: {prediccion[0]:.2f}")