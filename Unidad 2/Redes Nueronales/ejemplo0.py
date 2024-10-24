from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import confusion_matrix

cancer_data = load_breast_cancer()
x, y = cancer_data.data, cancer_data.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#estandariza los datos eliminado la media y escalando los datos
#de froma que su varianza sea igual 1
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

mlp = MLPClassifier(hidden_layer_sizes=(32,16,32,16), activation="tanh",max_iter=1000,random_state=42)

#entrenar modelo
mlp.fit(x_train, y_train)

#hacer oredicciones de los datos de prueba
y_pred = mlp.predict(x_test)
print(y_pred)

#Calcular la exactitud que modela
accuracy = accuracy_score(y_test,y_pred)
print(f"Accuracy: {accuracy:.2f}")

class_report = classification_report(y_test,y_pred)
print("Classification Report: \n", class_report)

conf_matrix = confusion_matrix(y_test,y_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = conf_matrix, display_labels = ['Benigno', 'Maligno'])
cm_display.plot()

