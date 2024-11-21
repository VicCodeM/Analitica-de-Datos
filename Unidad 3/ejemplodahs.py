import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
from sklearn.datasets import fetch_california_housing

# Cargar el dataset de California Housing
california_housing = fetch_california_housing(as_frame=True)
df = california_housing.frame

# Crear la aplicación Dash
app = dash.Dash(__name__)

# Diseño de la aplicación
app.layout = html.Div(children=[
    html.H1(children='Análisis de California Housing'),
    
    # Dropdown para seleccionar la característica X
    html.Label('Selecciona la característica X:'),
    dcc.Dropdown(
        id='x-axis',
        options=[{'label': col, 'value': col} for col in df.columns],
        value='MedInc'
    ),
    
    # Dropdown para seleccionar la característica Y
    html.Label('Selecciona la característica Y:'),
    dcc.Dropdown(
        id='y-axis',
        options=[{'label': col, 'value': col} for col in df.columns],
        value='HouseAge'
    ),
    
    # Gráfico de dispersión
    dcc.Graph(id='scatter-plot'),
    
    # Histograma
    dcc.Graph(id='histogram'),
    
    # Dropdown para seleccionar la característica para el histograma
    html.Label('Selecciona la característica para el histograma:'),
    dcc.Dropdown(
        id='histogram-feature',
        options=[{'label': col, 'value': col} for col in df.columns],
        value='MedInc'
    )
])

# Callback para actualizar el gráfico de dispersión
@app.callback(
    Output('scatter-plot', 'figure'),
    [Input('x-axis', 'value'),
     Input('y-axis', 'value')]
)
def update_scatter_plot(x_axis, y_axis):
    fig = px.scatter(df, x=x_axis, y=y_axis, title=f'{y_axis} vs {x_axis}')
    return fig

# Callback para actualizar el histograma
@app.callback(
    Output('histogram', 'figure'),
    [Input('histogram-feature', 'value')]
)
def update_histogram(feature):
    fig = px.histogram(df, x=feature, title=f'Distribución de {feature}')
    return fig

# Ejecutar la aplicación
if __name__ == '__main__':
    app.run_server(debug=True)