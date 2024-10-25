from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np

cal_housing = fetch_california_housing()
x,y = cal_housing.data, cal_housing.target

y_df = pd.DataFrame(y, columns=["price"])
y_df["price_category"] = pd.cut(y_df["price"], bins=[-float('inf'), 1.5, 3.0, float('inf')], labels=["Bajo", "Medio", "Alto"])
# Discretizar el objetivo (y) en categorías
# Por ejemplo, categorizamos en 3 clases: bajo, medio, alto.
#y_categorized = np.digitize(y, bins=[1.5, 3.0])  # Ajusta los valores según las necesidades de tu análisis

X_train, X_test, y_train, y_test = train_test_split(x, y_df["price_category"], test_size=0.2, random_state=1)


#Estandariza los datos eliminados la media y la escalando los datos
#de forma que su varianza sea igual a 1.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

mlp = MLPClassifier(hidden_layer_sizes=(256,128,64,32),activation="relu",max_iter=1000,random_state=42).fit(X_train,y_train)

#Entrenar el modelo
mlp.fit(X_train,y_train)

#Hacer las predicciones con los datos de prueba
y_pred = mlp.predict(X_test)
print(mlp.score(X_test,y_test))

#Calcular la exactitud del modela
accuracy = accuracy_score(y_test,y_pred)
print(f"Accuracy: {accuracy:.2f}")

class_report = classification_report(y_test,y_pred)
print("Classificaton Report:\n", class_report)

conf_matrix =  confusion_matrix(y_test,y_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=conf_matrix,display_labels=["Bajo","Medio","Alto"])
cm_display.plot()