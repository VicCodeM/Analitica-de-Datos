import pandas as pd
import numpy as np
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, learning_curve
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

# Calcular la curva de aprendizaje
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
            html.H5("Impacto del Precio Unitario en las Ventas Totales"),
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
            html.H5("Distribución del Precio Unitario o Total de Ventas"),
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
            html.H5("Promedio de Ventas por Cantidad o Rating"),
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
            html.H5("Curva de Aprendizaje del Modelo"),
            dcc.Graph(id='learning-curve')
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
], fluid=True)

# Callbacks para actualizar gráficos y predicciones
@app.callback(
    Output('scatter-features', 'figure'),
    Input('x-axis', 'value'),
    Input('y-axis', 'value')
)
def update_scatter(x_col, y_col):
    return px.scatter(
        data, x=x_col, y=y_col, 
        color='Rating',  # Usar 'Rating' para colorear los puntos
        title=f"{x_col} vs {y_col} con colores basados en Rating",
        labels={'x': x_col, 'y': y_col},
        template='plotly_white'
    )

@app.callback(
    Output('histogram', 'figure'),
    Input('hist-variable', 'value')
)
def update_histogram(variable):
    fig = px.histogram(
        data, x=variable, nbins=30, color_discrete_sequence=['#636EFA'],
        title=f"Distribución de {variable} para una mejor toma de decisiones",
        template='plotly_white'
    )
    fig.update_traces(marker_line_width=1, marker_line_color='black')
    return fig

@app.callback(
    Output('bar-chart', 'figure'),
    Input('bar-group', 'value')
)
def update_bar_chart(group_by):
    grouped_data = data.groupby(group_by)['Total'].mean().reset_index()
    fig = px.bar(
        grouped_data, x=group_by, y='Total',
        color='Total', color_continuous_scale='Viridis',
        title=f"Promedio de ventas por {group_by} (toma de decisiones)",
        template='plotly_white'
    )
    fig.update_layout(coloraxis_colorbar=dict(title="Promedio Ventas"))
    return fig

@app.callback(
    Output('learning-curve', 'figure'),
    Input('predict-btn', 'n_clicks')
)
def update_learning_curve(n_clicks):
    fig = px.line(
        x=train_sizes, y=[train_scores_mean, test_scores_mean],
        title="Curva de Aprendizaje (Evaluación del modelo)",
        labels={'x': 'Número de muestras de entrenamiento', 'y': 'Error cuadrático medio'},
        template='plotly_white'
    )
    fig.update_layout(
        legend=dict(title="Curvas"),
        legend_traceorder="normal",
        annotations=[dict(
            x=0.8, y=0.5, xref="paper", yref="paper",
            text="Modelo estabilizado, ajuste de parámetros necesario",
            showarrow=True, arrowhead=2, arrowsize=1
        )]
    )
    return fig

@app.callback(
    Output('prediction-output', 'children'),
    Input('unit-price', 'value'),
    Input('quantity', 'value'),
    Input('rating', 'value')
)
def update_prediction(unit_price, quantity, rating):
    # Realizar la predicción utilizando el modelo MLP
    input_data = scaler.transform([[unit_price, quantity, rating]])  # Escalar los valores de entrada
    prediction = mlp.predict(input_data)[0]  # Predecir el valor de ventas totales
    
    # Mostrar la predicción
    return f"El total estimado de ventas es: ${prediction:.2f}"

if __name__ == "__main__":
    app.run_server(debug=True)
