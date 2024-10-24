from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt

iris_data = load_iris()
x= iris_data.data
y= iris_data.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
sc_x= StandardScaler()
x_train= sc_x.fit_transform(x_train)
x_test= sc_x.transform(x_test)

clf = MLPClassifier(hidden_layer_sizes=(256,128,64,32), activation="relu",max_iter=1000,random_state=42).fit(x_train, y_train)


#entrenar modelo
clf.fit(x_train, y_train)

#hacer oredicciones de los datos de prueba
y_pred = clf.predict(x_test)
print(clf.score(x_test, y_test))

#Calcular la exactitud que modela
accuracy = accuracy_score(y_test,y_pred)
print(f"Accuracy: {accuracy:.2f}")

class_report = classification_report(y_test,y_pred)
print("Classification Report: \n", class_report)

conf_matrix = confusion_matrix(y_test,y_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = conf_matrix, display_labels = ['Setosa', 'Versicolor', 'Virginica'])
cm_display.plot()
