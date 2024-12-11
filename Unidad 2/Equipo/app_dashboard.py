import pandas as pd
import numpy as np
from dash1 import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Configurar tema de Bootstrap
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Paso 1: Cargar y preprocesar el dataset
data = pd.read_csv('salesproject.csv')
columnas_a_eliminar = ['Invoice ID', 'City', 'Customer type', 'Date', 'Time', 'Payment', 'cogs', 
                       'gross margin percentage', 'gross income', 'Tax 5%']
data = data.drop(columns=columnas_a_eliminar)
features = ['Unit price', 'Quantity', 'Rating']
X = data[features]
y = data['Total']

# Dividir el dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Modelo de red neuronal
mlp = MLPRegressor(hidden_layer_sizes=(30, 30), max_iter=5000, alpha=0.1, early_stopping=True, random_state=42)
mlp.fit(X_train, y_train)
y_pred_nn = mlp.predict(X_test)

# Métricas del modelo
rmse_nn = mean_squared_error(y_test, y_pred_nn, squared=False)
r2_nn = r2_score(y_test, y_pred_nn)

# Crear datos para curva de aprendizaje
from sklearn.model_selection import learning_curve
train_sizes, train_scores, test_scores = learning_curve(
    mlp, X_train, y_train, cv=5, scoring='neg_mean_squared_error', train_sizes=np.linspace(0.1, 1.0, 10)
)
train_scores_mean = -train_scores.mean(axis=1)
test_scores_mean = -test_scores.mean(axis=1)

# Layout del dashboard
app.layout = dbc.Container([
    html.H1("Dashboard Interactivo: Predicción de Ventas", className="text-center my-4"),
    
    # Métricas del modelo
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("RMSE del modelo", className="card-title"),
                    html.P(f"{rmse_nn:.2f}", className="card-text text-center"),
                ])
            ], className="shadow-sm"),
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("R² del modelo", className="card-title"),
                    html.P(f"{r2_nn:.2f}", className="card-text text-center"),
                ])
            ], className="shadow-sm"),
        ], width=6)
    ], className="my-4"),
    
    # Gráficos interactivos
    dbc.Row([
        # Gráfico de dispersión
        dbc.Col([
            html.H5("Relación entre variables"),
            dcc.Dropdown(
                id='x-axis',
                options=[{'label': col, 'value': col} for col in features + ['Total']],
                value='Unit price',
                placeholder="Selecciona el eje X",
                className="mb-2"
            ),
            dcc.Dropdown(
                id='y-axis',
                options=[{'label': col, 'value': col} for col in features + ['Total']],
                value='Total',
                placeholder="Selecciona el eje Y",
                className="mb-2"
            ),
            dcc.Graph(id='scatter-features')
        ], width=6),
        
        # Histograma
        dbc.Col([
            html.H5("Distribución de datos"),
            dcc.Dropdown(
                id='hist-variable',
                options=[{'label': col, 'value': col} for col in features + ['Total']],
                value='Total',
                placeholder="Selecciona una variable",
                className="mb-2"
            ),
            dcc.Graph(id='histogram')
        ], width=6),
    ], className="my-4"),
    
    dbc.Row([
        # Gráfico de barras
        dbc.Col([
            html.H5("Total de ventas agrupado"),
            dcc.Dropdown(
                id='bar-group',
                options=[{'label': col, 'value': col} for col in ['Quantity', 'Rating']],
                value='Quantity',
                placeholder="Selecciona agrupación",
                className="mb-2"
            ),
            dcc.Graph(id='bar-chart')
        ], width=6),
        
        # Curva de aprendizaje
        dbc.Col([
            html.H5("Curva de aprendizaje"),
            dcc.Graph(
                id='learning-curve',
                figure=px.line(
                    x=train_sizes,
                    y=[train_scores_mean, test_scores_mean],
                    labels={'x': 'Tamaño del conjunto de entrenamiento', 'y': 'Error'},
                    title="Curva de aprendizaje",
                ).update_traces(name='Entrenamiento').add_scatter(
                    x=train_sizes, y=test_scores_mean, mode='lines', name='Prueba'
                )
            )
        ], width=6),
    ], className="my-4"),
    
    # Predicción interactiva
    html.H5("Predicción interactiva"),
    dbc.Row([
        dbc.Col([
            dbc.InputGroup([
                dbc.InputGroupText("Unit Price"),
                dbc.Input(id='unit-price', type='number', value=2000, step=0.1)
            ], className="mb-2"),
            dbc.InputGroup([
                dbc.InputGroupText("Quantity"),
                dbc.Input(id='quantity', type='number', value=9, step=1)
            ], className="mb-2"),
            dbc.InputGroup([
                dbc.InputGroupText("Rating"),
                dbc.Input(id='rating', type='number', value=0.2, step=0.1)
            ], className="mb-2"),
            dbc.Button("Predecir", id='predict-btn', n_clicks=0, color="primary", className="mt-2"),
        ], width=4),
        dbc.Col([
            html.Div(id='prediction-output', className="alert alert-info", style={'fontSize': '18px'})
        ], width=8),
    ], className="my-4"),
    
    # Explicación de gráficos
    html.Hr(),
    html.Div([
        html.H4("Explicaciones de los Gráficos"),
        html.Ul([
            html.Li("Relación entre variables: Permite visualizar cómo se relacionan dos variables, como el precio unitario y el total de ventas."),
            html.Li("Distribución de datos: Muestra la frecuencia de valores en una variable específica, útil para detectar patrones."),
            html.Li("Total de ventas agrupado: Presenta el total de ventas agrupadas por una característica, como la cantidad o la calificación."),
            html.Li("Curva de aprendizaje: Analiza cómo el error del modelo varía con diferentes tamaños de datos de entrenamiento."),
        ])
    ], className="my-4")
], fluid=True)

# Callbacks para actualizar gráficos y predicciones
@app.callback(
    Output('scatter-features', 'figure'),
    Input('x-axis', 'value'),
    Input('y-axis', 'value')
)
def update_scatter(x_col, y_col):
    return px.scatter(data, x=x_col, y=y_col, title=f"{x_col} vs {y_col}")

@app.callback(
    Output('histogram', 'figure'),
    Input('hist-variable', 'value')
)
def update_histogram(variable):
    return px.histogram(data, x=variable, nbins=30, title=f"Distribución de {variable}")

@app.callback(
    Output('bar-chart', 'figure'),
    Input('bar-group', 'value')
)
def update_bar_chart(group_by):
    grouped_data = data.groupby(group_by)['Total'].sum().reset_index()
    return px.bar(grouped_data, x=group_by, y='Total', title=f"Total de ventas por {group_by}")

@app.callback(
    Output('prediction-output', 'children'),
    Input('unit-price', 'value'),
    Input('quantity', 'value'),
    Input('rating', 'value'),
    Input('predict-btn', 'n_clicks')
)
def update_prediction(unit_price, quantity, rating, n_clicks):
    nuevos_datos = np.array([[unit_price, quantity, rating]])
    nuevos_datos = scaler.transform(nuevos_datos)
    prediccion_nn = mlp.predict(nuevos_datos)
    return f"Predicción del total de ventas: {prediccion_nn[0]:.2f}"

# Ejecutar servidor
if __name__ == '__main__':
    app.run_server(debug=True)
