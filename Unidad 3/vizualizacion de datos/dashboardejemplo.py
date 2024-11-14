import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QTableWidget, QTableWidgetItem, QPushButton, QTabWidget
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Configurar la variable de entorno para usar PySide6
os.environ['QT_API'] = 'pyside6'

# Datos de ejemplo
data = {
    'Nombre': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'Edad': [24, 27, 22, 32, 29],
    'Salario': [50000, 60000, 45000, 80000, 75000]
}
df = pd.DataFrame(data)

class Dashboard(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Dashboard de Minería de Datos")
        self.setGeometry(100, 100, 800, 600)

        # Widget principal
        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)

        # Layout principal
        self.layout = QVBoxLayout(self.main_widget)

        # Pestañas
        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs)

        # Pestaña 1: Tabla de Datos
        self.tab1 = QWidget()
        self.tabs.addTab(self.tab1, "Tabla de Datos")
        self.tab1_layout = QVBoxLayout(self.tab1)

        self.table = QTableWidget(self.tab1)
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(['Nombre', 'Edad', 'Salario'])
        self.tab1_layout.addWidget(self.table)

        self.update_button = QPushButton("Actualizar Tabla", self.tab1)
        self.update_button.clicked.connect(self.update_table)
        self.tab1_layout.addWidget(self.update_button)

        # Pestaña 2: Gráfico
        self.tab2 = QWidget()
        self.tabs.addTab(self.tab2, "Gráfico")
        self.tab2_layout = QVBoxLayout(self.tab2)

        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.tab2_layout.addWidget(self.canvas)

        self.plot_button = QPushButton("Actualizar Gráfico", self.tab2)
        self.plot_button.clicked.connect(self.update_plot)
        self.tab2_layout.addWidget(self.plot_button)

        # Pestaña 3: Red Neuronal
        self.tab3 = QWidget()
        self.tabs.addTab(self.tab3, "Red Neuronal")
        self.tab3_layout = QVBoxLayout(self.tab3)

        self.nn_button = QPushButton("Entrenar Red Neuronal", self.tab3)
        self.nn_button.clicked.connect(self.train_neural_network)
        self.tab3_layout.addWidget(self.nn_button)

        # Inicializar la tabla y el gráfico
        self.update_table()
        self.update_plot()

    def update_table(self):
        self.table.setRowCount(len(df))
        for i, row in df.iterrows():
            self.table.setItem(i, 0, QTableWidgetItem(str(row['Nombre'])))
            self.table.setItem(i, 1, QTableWidgetItem(str(row['Edad'])))
            self.table.setItem(i, 2, QTableWidgetItem(str(row['Salario'])))

    def update_plot(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        df.plot(kind='bar', x='Nombre', y='Salario', ax=ax)
        self.canvas.draw()

    def train_neural_network(self):
        # Datos de ejemplo para la red neuronal
        X = df[['Edad']]
        y = df['Salario'] > 60000  # Clasificación binaria

        # Dividir los datos en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Crear y entrenar la red neuronal
        clf = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=42)
        clf.fit(X_train, y_train)

        # Predecir y calcular la precisión
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"Precisión de la red neuronal: {accuracy:.2f}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Dashboard()
    window.show()
    sys.exit(app.exec())