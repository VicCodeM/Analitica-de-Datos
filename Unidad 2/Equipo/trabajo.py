import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn import metrics
import matplotlib.pyplot as plt

# Cargar el dataset
df = pd.read_csv('trainpre (1).csv')

# Separar las características y la variable objetivo
X = df.drop('Sales', axis=1)
y = df['Sales']

# Identificar las columnas categóricas y numéricas
categorical_cols = X.select_dtypes(include=['object']).columns
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns

# Preprocesamiento: One-hot encoding para las columnas categóricas y escalado para las numéricas
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)  # Configurar para ignorar categorías desconocidas
    ])

# Definir el modelo de red neuronal para regresión
mlp = MLPRegressor(hidden_layer_sizes=(256, 128, 64, 32), activation="relu", max_iter=1000, random_state=42)

# Crear un pipeline que primero preprocesa los datos y luego entrena el modelo
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', mlp)
])

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Entrenar el modelo
pipeline.fit(X_train, y_train)

# Hacer las predicciones con los datos de prueba
y_pred = pipeline.predict(X_test)

# Calcular métricas de regresión
mse = metrics.mean_squared_error(y_test, y_pred)
r2 = metrics.r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Mostrar las primeras predicciones y los valores reales
print("Predicciones:", y_pred[:5]) 
print("Valores reales:", y_test[:5]) 

# Gráfico 1: Diagrama de dispersión de Predicciones vs Valores Reales
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color="blue")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Valores Reales')
plt.ylabel('Predicciones')
plt.title('Predicciones vs Valores Reales')
plt.show()