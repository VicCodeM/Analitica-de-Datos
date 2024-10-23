from sklearn.naive_bayes import GaussianNB


#entrenamiento del algoritmo
modelo_nb = GaussianNB()

modelo_nb.fit(x_entrenamiento, y_entrenamiento)

# Predicciones del algritmo  para procesameinti de datos de prueba
predicion = modelo_nb.predict(x_prueba)