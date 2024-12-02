import dash
from dash import dash_table
import pandas as pd
from dash import dcc, html
from dash import dash_table
import plotly.graph_objs as go
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import dash
import dash
import pickle
import plotly.graph_objects as go
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
from dash import dcc, html
from dash import dash_table
import plotly.graph_objs as go
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf
import io
import base64
from dash.dependencies import Input, Output, State
import numpy as np
from dash import dcc, html
import dash_table
import plotly.graph_objs as go

# Configuración de Dash
app = dash.Dash(
    external_stylesheets=[dbc.themes.MINTY],
    suppress_callback_exceptions=True
)
server = app.server

# Cargar datos
try:
    file_path = "ws.xlsx"
    df = pd.read_excel(file_path)
except Exception as e:
    df = pd.DataFrame()
    print(f"Error al cargar los datos: {e}")

if not df.empty and 'Fecha' in df.columns:
    df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
    df = df.set_index('Fecha')

variables_a_analizar = [col for col in df.columns if df[col].dtype in ['float64', 'int64']]
missing_data = df.isnull().mean() * 100 if not df.empty else pd.Series()

# Función para crear gráficos de autocorrelación
def plot_autocorrelation(series):
    acf_values = acf(series.dropna(), nlags=20)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(range(len(acf_values))), y=acf_values, marker_color='#004085'))
    fig.update_layout(title="Autocorrelación", xaxis_title="Lags", yaxis_title="ACF")
    return fig

# Función para crear gráficos de correlación parcial
def plot_partial_autocorrelation(series):
    pacf_values = pacf(series.dropna(), nlags=20)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(range(len(pacf_values))), y=pacf_values, marker_color='#004085'))
    fig.update_layout(title="Correlación Parcial", xaxis_title="Lags", yaxis_title="PACF")
    return fig

layout_pagina_eda = html.Div([
    html.H1('Análisis Exploratorio de Datos', style={
        'textAlign': 'center', 'color': '#004085', 'fontSize': '24px', 'fontFamily': 'Georgia, serif'}),
    dcc.Tabs(id='tabs-eda', value='tab-general', className='custom-tabs', children=[
        dcc.Tab(label='General', value='tab-general', className='custom-tab', selected_className='custom-tab--selected'),
        dcc.Tab(label='Univariado', value='tab-univariado', className='custom-tab', selected_className='custom-tab--selected'),
        dcc.Tab(label='Bivariado', value='tab-bivariado', className='custom-tab', selected_className='custom-tab--selected'),
    ]),
    html.Div(id='tabs-eda-content', style={'marginTop': '20px'})
])


app.index_string += '''
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap" rel="stylesheet">
<style>
body {
    font-family: 'Poppins', sans-serif;
    font-size: 16px;
}
.custom-tabs {
    font-family: 'Poppins', sans-serif;
    background-color: #004085;
    border-radius: 8px;
    padding: 5px;
}
.custom-tab {
    color: grey;
    font-weight: 600; /* Semi-bold */
    padding: 8px;
    border-radius: 5px;
    margin-right: 5px;
    transition: background-color 0.3s ease, color 0.3s ease;
    text-align: center;
    font-size: 14px;
}
.custom-tab:hover {
    background-color: #0062cc;
    color: #004085;
}
.custom-tab--selected {
    background-color: white;
    color: #004085;
    font-weight: 700; /* Bold */
    border: 2px solid #004085;
    text-align: center;
}
.info-box {
    background-color: #004085;
    color: white;
    border-radius: 10px;
    padding: 15px;
    text-align: center;
    font-weight: 600; /* Semi-bold */
    font-size: 16px;
}
.info-title {
    font-size: 16px;
    margin-bottom: 5px;
    font-family: 'Poppins', sans-serif;
    font-weight: 600; /* Semi-bold */
}
.info-value {
    font-size: 20px;
    font-family: 'Poppins', sans-serif;
    font-weight: 700; /* Bold */
}
h1 {
    font-family: 'Poppins', sans-serif;
    font-size: 24px;
    font-weight: 700; /* Bold */
    color: #004085;
}
h3 {
    font-family: 'Poppins', sans-serif;
    font-size: 20px;
    font-weight: 600; /* Semi-bold */
    color: #004085;
    margin-top: 20px;
    text-align: center;
}
h5 {
    font-family: 'Poppins', sans-serif;
    font-size: 18px;
    font-weight: 600; /* Semi-bold */
    color: #004085;
}

.dropdown-custom {
    border: 1px solid #004085; /* Borde azul oscuro */
    border-radius: 8px; /* Bordes redondeados */
    padding: 10px; /* Espaciado interno */
    font-size: 16px; /* Tamaño de fuente */
    font-family: 'Poppins', sans-serif; /* Mismo estilo que los tabs */
    font-weight: 600; /* Semi-bold para consistencia */
    background-color: #f8f9fa; /* Fondo claro */
    color: #004085; /* Texto en azul oscuro */
    width: 100%; /* Ocupa todo el espacio del contenedor */
    box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1); /* Sombra ligera */
    transition: border-color 0.3s ease, box-shadow 0.3s ease; /* Transiciones */
}

/* Estilo al hacer hover */
.dropdown-custom:hover {
    border-color: #0062cc; /* Azul más vibrante */
    box-shadow: 0px 6px 8px rgba(0, 0, 0, 0.2); /* Sombra más intensa */
}

/* Estilo cuando está enfocado */
.dropdown-custom:focus {
    border-color: #80bdff; /* Azul claro */
    outline: none; /* Sin borde adicional */
    box-shadow: 0 0 0 3px rgba(128, 189, 255, 0.5); /* Resaltado */
}

/* Estilo de las opciones del menú desplegable */
.dropdown-custom .dropdown-menu {
    border: 1px solid #004085;
    border-radius: 8px;
    background-color: #ffffff; /* Fondo blanco */
    font-family: 'Poppins', sans-serif; /* Mismo estilo */
    font-weight: 600; /* Semi-bold */
    color: #004085;
}
</style>
'''
# Layout de las páginas
layout_pagina_inicio = html.Div([
    html.H1('De vientos y datos', style={'textAlign': 'center', 'fontWeight': 'bold', 'fontSize': '3rem', 'color': '#004F6D', 'textShadow': '1px 1px 2px #000000'}),
    dcc.Tabs(id='tabs-intro', value='tab-intro', children=[
        dcc.Tab(label='Introducción', value='tab-int', style={'border': '2px solid #004F6D', 'color': '#004F6D', 'fontWeight': 'bold'}),
        dcc.Tab(label='Descripción de los datos', value='tab-descripcion', style={'border': '2px solid #004F6D', 'color': '#004F6D', 'fontWeight': 'bold'}),
        dcc.Tab(label='Diccionario de variables', value='tab-diccionario', style={'border': '2px solid #004F6D', 'color': '#004F6D', 'fontWeight': 'bold'}),
    ]),
    html.Div(id='tabs-intro-content')
])

# Diccionario de variables
tabla = {
    'Variable': [
        'Fecha', 'WindSpeed100m_1', 'WinSpeed100m_2', 
        'WindSpeed80m_1', 'WindSpeed80m_2', 'WindSpeed60m', 'DirViento60m', 
        'DirViento100m', 'DirViento80m',
        'Presion', 'Humedad', 'Temp100m', 
        'Temp21m'
    ],
    'Descripción': [
        'Fecha de la medición', 
        'Velocidad del viento a 100 metros, primer sensor', 
        'Velocidad del viento a 100 metros, segundo sensor', 
        'Velocidad del viento a 80 metros, primer sensor', 
        'Velocidad del viento a 80 metros, segundo sensor', 
        'Velocidad del viento a 60 metros', 
        'Dirección del viento a 60 metros',
        'Dirección del viento a 100 metros',
        'Dirección del viento a 80 metros',
        'Presión atmosférica en la ubicación de la medición', 
        'Humedad relativa (%) en la ubicación de la medición', 
        'Temperatura a 100 metros de altura', 
        'Temperatura a 21 metros de altura'
    ],
    'Unidad de Medida': [
        '-', 'm/s', 'm/s', 'm/s', 'm/s', 'm/s', 
        'Grados (°)', 'Grados (°)', 'Grados (°)',
        'hPa', '%', '°C', '°C'
    ]
}

tabla_df = pd.DataFrame(tabla)

# Callback para mostrar el contenido en las pestañas de introducción
@app.callback(
    dash.dependencies.Output('tabs-intro-content', 'children'),
    [dash.dependencies.Input('tabs-intro', 'value')]
)
def tab_layout1(tab):
    if tab == 'tab-int':
        return html.Div([
            html.P([
                'Un sistema de monitoreo instalado en una región costera recoge datos cada 10 minutos sobre variables como ',
                html.B('velocidad y dirección del viento, temperatura, presión atmosférica y humedad relativa'),
                ', con el objetivo de evaluar la viabilidad de la generación de energía eólica. Estos sensores son fundamentales para ',
                html.B('cuantificar recursos eólicos y monitorear condiciones climáticas'),
                '. Este proyecto, liderado por ',
                html.B('Emanuel Carbonell y Kanery Camargo'),
                ', utiliza técnicas de ',
                html.B('Machine Learning (Modelos benchmark, Modelos originales)'),
                ' para analizar estas series de tiempo, identificar patrones y optimizar la predicción de energía eólica. Además, se desarrollarán visualizaciones interactivas para facilitar la toma de decisiones estratégicas basadas en estos datos.'
            ], style={'margin-top': '40px', 'text-align': 'justify'}),
            html.Div([
                html.Img(src='/assets/intro2.png', style={'width': '34%', 'height': '340px', 'margin-right': '4%'}),
                html.Img(src='/assets/intro.png', style={'width': '34%', 'height': '340px'})
            ], style={'display': 'flex', 'justify-content': 'center', 'margin-top': '20px'})
        ])

    elif tab == 'tab-descripcion':
        return html.Div([
            html.Div([
                html.P("""
                    Este proyecto utiliza un conjunto de datos meteorológicos capturados a través de sensores instalados a diferentes alturas
                    (40 m, 60 m, 80 m y 100 m) para analizar las condiciones atmosféricas en una región costera. El objetivo principal con este conjunto de datos es identificar patrones clave y utilizar técnicas avanzadas de Machine Learning para optimizar la predicción de la energía eólica.
                """, style={'text-align': 'justify', 'margin-bottom': '20px', 'margin-top': '20px'})
            ]),
            html.Div([
                html.Div([
                    html.Img(src='/assets/desc1.png', style={'width': '180px', 'height': '180px', 'margin-bottom': '20px'}),
                    html.Img(src='/assets/desc2.png', style={'width': '180px', 'height': '180px', 'margin-bottom': '20px'})
                ], style={'display': 'flex', 'flex-direction': 'column', 'justify-content': 'flex-start', 'margin-right': '20px'}),
                html.Div([
                    dbc.Button('Variables Clave', id='collapse-button-1', className='mb-3', color='primary',
                               style={'background-color': '#004F6D', 'width': '250px', 'font-size': '16px', 'margin-right': '10px'}),
                    dbc.Button('Reducción de Dimensionalidad', id='collapse-button-2', className='mb-3', color='primary',
                               style={'background-color': '#004F6D', 'width': '250px', 'font-size': '16px', 'margin-right': '10px'}),
                    dbc.Button('Agrupación de los Datos', id='collapse-button-3', className='mb-3', color='primary',
                               style={'background-color': '#004F6D', 'width': '250px', 'font-size': '16px', 'margin-right': '10px'})
                ], style={'display': 'flex', 'flex-direction': 'column', 'justify-content': 'center', 'align-items': 'center'}),
                html.Div([
                    html.Img(src='/assets/desc3.png', style={'width': '180px', 'height': '180px', 'margin-bottom': '10px'}),
                    html.Img(src='/assets/desc4.png', style={'width': '180px', 'height': '180px', 'margin-bottom': '10px'})
                ], style={'display': 'flex', 'flex-direction': 'column', 'justify-content': 'flex-start', 'margin-left': '20px'})
            ], style={'display': 'flex', 'justify-content': 'center', 'align-items': 'center', 'margin-top': '40px'}),
            dbc.Collapse(
                html.Div([
                    html.Ul([
                        html.Li("Velocidad del viento (m/s), medida en cada una de las alturas."),
                        html.Li("Dirección del viento (grados), correspondiente a la orientación en cada nivel."),
                        html.Li("Temperatura del aire (°C), registrada a 20 m y 100 m."),
                        html.Li("Presión atmosférica (hPa) y humedad relativa (%), medidas por los sensores.")
                    ], style={'textAlign': 'left', 'lineHeight': '1.8'}),
                ], style={
                    'background-color': '#f0f0f0', 
                    'padding': '15px', 
                    'border-radius': '15px', 
                    'width': '60%', 
                    'margin': '20px auto'
                }),
                id='collapse-1', is_open=False
            ),
            dbc.Collapse(
                html.Div([
                    html.P("""
                        Se usaron únicamente las columnas correspondientes a el promedio de cada variable.
                    """, style={'text-align': 'justify'}),
                ], style={'background-color': '#f0f0f0', 'padding': '15px', 'border-radius': '15px', 'width': '60%', 'margin': '20px auto'}),
                id='collapse-2', is_open=False
            ),
            dbc.Collapse(
                html.Div([
                    html.P("""
                        Los datos se agruparon por la altura de los sensores (40 m, 60 m, 80 m, 100 m)
                        para un análisis más preciso de las condiciones atmosféricas en distintos niveles. 
                        Esto permite evaluar cómo varían el viento y la temperatura, lo cual es clave para la viabilidad de proyectos eólicos.
                        El agrupamiento también facilita la visualización y el análisis de patrones, mejorando la interpretación de los datos.
                    """, style={'text-align': 'justify'}),
                ], style={'background-color': '#f0f0f0', 'padding': '15px', 'border-radius': '15px', 'width': '60%', 'margin': '20px auto'}),
                id='collapse-3', is_open=False
            )
        ])

    elif tab == 'tab-diccionario':
        return html.Div([
            html.P("""
                A continuación, se presenta un diccionario de variables con una breve descripción de cada una, esto es importante para entender el significado de cada columna en el conjunto de datos.:
            """, style={'text-align': 'justify', 'margin-bottom': '20px', 'margin-top': '20px'}),
            html.Div([
                dash_table.DataTable(
                    id='table',
                    columns=[{"name": i, "id": i} for i in tabla_df.columns],
                    data= tabla_df.to_dict('records'),
                    style_table={'width': '600px', 'overflowX': 'scroll'},  # Ancho fijo con scroll horizontal
                    style_cell={
                        'textAlign': 'left',
                        'padding': '10px',
                        'fontFamily': 'Arial',
                        'whiteSpace': 'normal',  # Ajustar contenido de la celda
                        'height': 'auto',  # Permitir altura automática
                        'maxWidth': '300px',  # Limitar el ancho de las celdas
                        'overflow': 'hidden',  # Ocultar el desbordamiento
                        'textOverflow': 'ellipsis',  # Recortar texto largo
                        'border': '2px solid #004F6D'
                    },
                    style_header={
                        'backgroundColor': '#004F6D',  # Fondo azul oscuro
                        'fontWeight': 'bold',
                        'textAlign': 'center',
                        'color': 'white',  # Cambiar el color de la letra del encabezado
                        'border': '2px solid #004F6D'  # Borde en el encabezado
                    },
                )
            ], style={'width': '600px', 'margin': '0 auto', 'padding': '20px'})  # Contenedor de ancho fijo
        ], style={'display': 'block', 'align-items': 'center', 'height': '100vh'})

# Callback para manejar los botones de colapso permitiendo solo uno abierto a la vez
@app.callback(
    [dash.dependencies.Output(f'collapse-{i}', 'is_open') for i in range(1, 4)],
    [dash.dependencies.Input(f'collapse-button-{i}', 'n_clicks') for i in range(1, 4)],
    [dash.dependencies.State(f'collapse-{i}', 'is_open') for i in range(1, 4)]
)
def toggle_collapse(n1, n2, n3, is_open1, is_open2, is_open3):
    # Permitir que solo una sección esté abierta a la vez
    ctx = dash.callback_context

    if not ctx.triggered:
        return [False, False, False]
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'collapse-button-1':
        return [not is_open1, False, False]
    elif button_id == 'collapse-button-2':
        return [False, not is_open2, False]
    elif button_id == 'collapse-button-3':
        return [False, False, not is_open3]

    return [False, False, False]


# Layout EDA
layout_pagina_eda = html.Div([
    html.H1('Análisis Exploratorio de Datos', style={
        'textAlign': 'center', 'color': '#004085', 'fontSize': '24px', 'fontFamily': 'Poppins, sans-serif'}),
    dcc.Tabs(id='tabs-eda', value='tab-general', className='custom-tabs', children=[
        dcc.Tab(label='General', value='tab-general', className='custom-tab', selected_className='custom-tab--selected'),
        dcc.Tab(label='Univariado', value='tab-univariado', className='custom-tab', selected_className='custom-tab--selected'),
        dcc.Tab(label='Bivariado', value='tab-bivariado', className='custom-tab', selected_className='custom-tab--selected'),
    ]),
    html.Div([
        # Dropdown inicializado con la primera variable
        html.Div([
            html.Label('Selecciona una variable:', style={'fontWeight': 'bold', 'color': '#004085'}),
            dcc.Dropdown(
                id='variable-dropdown',
                options=[{'label': var, 'value': var} for var in variables_a_analizar],
                value=variables_a_analizar[0] if variables_a_analizar else None,
    className="dropdown-custom"
            )
        ], style={'marginBottom': '20px'}),
        html.Div(id='tabs-eda-content', style={'marginTop': '20px'})  # Contenido dinámico
    ])
])

@app.callback(
    Output('correlation-matrix', 'figure'),
    [Input('tabs-eda', 'value')]
)
def update_correlation_matrix(tab):
    if tab == 'tab-bivariado':
        # Cálculo de la matriz de correlación
        corr_matrix = df[variables_a_analizar].corr()
        fig = go.Figure(
            data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale='Blues'
            )
        )
        fig.update_layout(
            title='Matriz de Correlación',
            xaxis_title='Variables',
            yaxis_title='Variables',
            template='plotly_white'
        )
        return fig
    return {}

@app.callback(
    [Output('scatter-plot', 'figure'), Output('regplot', 'figure')],
    [Input('dropdown-x', 'value'), Input('dropdown-y', 'value')]
)
def update_bivariate_graphs(var_x, var_y):
    if var_x and var_y and var_x in df.columns and var_y in df.columns:
        # Scatterplot
        scatter_fig = go.Figure(
            data=go.Scatter(
                x=df[var_x],
                y=df[var_y],
                mode='markers',
                marker=dict(color='#004085', size=7, opacity=0.7)
            )
        )
        scatter_fig.update_layout(
            title=f'Scatterplot: {var_x} vs {var_y}',
            xaxis_title=var_x,
            yaxis_title=var_y,
            template='plotly_white'
        )

        # Regplot (línea de regresión)
        reg_fig = go.Figure(
            data=[
                go.Scatter(
                    x=df[var_x],
                    y=df[var_y],
                    mode='markers',
                    marker=dict(color='rgba(0, 64, 133, 0.3)', size=7)
                )
            ]
        )
        # Ajustar línea de regresión
        coef = np.polyfit(df[var_x], df[var_y], 1)
        reg_fig.add_trace(
            go.Scatter(
                x=df[var_x],
                y=coef[0] * df[var_x] + coef[1],
                mode='lines',
                line=dict(color='red', width=2),
                name='Línea de Regresión'
            )
        )
        reg_fig.update_layout(
            title=f'Regplot: {var_x} vs {var_y}',
            xaxis_title=var_x,
            yaxis_title=var_y,
            template='plotly_white'
        )
        return scatter_fig, reg_fig
    return {}, {}@app.callback(
    [Output('scatter-plot', 'figure'), Output('regplot', 'figure')],
    [Input('dropdown-x', 'value'), Input('dropdown-y', 'value')]
)
def update_bivariate_graphs(var_x, var_y):
    if var_x and var_y and var_x in df.columns and var_y in df.columns:
        # Scatterplot
        scatter_fig = go.Figure(
            data=go.Scatter(
                x=df[var_x],
                y=df[var_y],
                mode='markers',
                marker=dict(color='#004085', size=7, opacity=0.7)
            )
        )
        scatter_fig.update_layout(
            title=f'Scatterplot: {var_x} vs {var_y}',
            xaxis_title=var_x,
            yaxis_title=var_y,
            template='plotly_white'
        )

        # Regplot (línea de regresión)
        reg_fig = go.Figure(
            data=[
                go.Scatter(
                    x=df[var_x],
                    y=df[var_y],
                    mode='markers',
                    marker=dict(color='rgba(0, 64, 133, 0.3)', size=7)
                )
            ]
        )
        # Ajustar línea de regresión
        coef = np.polyfit(df[var_x], df[var_y], 1)
        reg_fig.add_trace(
            go.Scatter(
                x=df[var_x],
                y=coef[0] * df[var_x] + coef[1],
                mode='lines',
                line=dict(color='red', width=2),
                name='Línea de Regresión'
            )
        )
        reg_fig.update_layout(
            title=f'Regplot: {var_x} vs {var_y}',
            xaxis_title=var_x,
            yaxis_title=var_y,
            template='plotly_white'
        )
        return scatter_fig, reg_fig
    return {}, {}



@app.callback(
    Output('tabs-eda-content', 'children'),
    [Input('tabs-eda', 'value'),
     Input('variable-dropdown', 'value')]
)
def render_tabs_eda(tab, variable):
    if tab == 'tab-general':
        # Contenido para la pestaña "General"
        return html.Div([
            dbc.Row([
                dbc.Col(html.Div([
                    html.Div("Número de Observaciones", className="info-title"),
                    html.Div(f"{df.shape[0]}" if not df.empty else "0", className="info-value")
                ], className="info-box"), width=5),
                dbc.Col(html.Div([
                    html.Div("Número de Variables", className="info-title"),
                    html.Div(f"{df.shape[1]}" if not df.empty else "0", className="info-value")
                ], className="info-box"), width=5),
            ], justify='center', style={'marginTop': '20px'}),
            html.H3('Resumen de Variables', style={'marginTop': '20px', 'color': '#004085'}),
            dash_table.DataTable(
                id='tabla-summary',
                columns=[{'name': col, 'id': col} for col in ['Variable'] + list(df.describe().index)],
                data=[{'Variable': var} | df[var].describe().round(2).to_dict() for var in variables_a_analizar] if not df.empty else [],
                style_table={'overflowX': 'auto', 'width': '90%', 'margin': '0 auto'},
                style_cell={
                    'textAlign': 'center',
                    'padding': '10px',
                    'fontSize': '14px',
                    'backgroundColor': '#f8f9fa',
                    'color': '#000'
                },
                style_header={
                    'backgroundColor': '#004085',
                    'color': 'white',
                    'fontWeight': 'bold',
                    'border': '1px solid white'
                },
                style_data={
                    'border': '1px solid #004085'
                }
            )
        ])
    elif tab == 'tab-univariado':
        # Contenido para la pestaña "Univariado"
        if variable and variable in df.columns:
            # Descomposición de la serie
            decomposition = seasonal_decompose(df[variable], model='additive', period=12)
            observed = decomposition.observed
            trend = decomposition.trend
            seasonal = decomposition.seasonal
            residual = decomposition.resid

            # Gráfico de Descomposición de la Serie Temporal
            decomposition_fig = go.Figure()
            decomposition_fig.add_trace(go.Scatter(
                x=df.index, y=observed, mode='lines', name='Observed', line=dict(color='#004085'), visible=True))  # Azul oscuro
            decomposition_fig.add_trace(go.Scatter(
                x=df.index, y=trend, mode='lines', name='Trend', line=dict(color='#66B2FF'), visible='legendonly'))  # Azul claro
            decomposition_fig.add_trace(go.Scatter(
                x=df.index, y=seasonal, mode='lines', name='Seasonal', line=dict(color='#00CC99'), visible='legendonly'))  # Verde
            decomposition_fig.add_trace(go.Scatter(
                x=df.index, y=residual, mode='lines', name='Residual', line=dict(color='#FF9933'), visible='legendonly'))  # Naranja
            decomposition_fig.update_layout(
                title="Descomposición de la Serie",
                xaxis_title="Fecha",
                yaxis_title=variable,
                template="plotly_white",
                legend=dict(
                    x=0.01, y=0.99, bgcolor='rgba(255,255,255,0)',
                    bordercolor='rgba(255,255,255,0)'
                )
            )

            # Histograma
            histogram_fig = go.Figure(data=[
                go.Histogram(x=df[variable], nbinsx=30, marker_color='#004085')
            ])
            histogram_fig.update_layout(
                title="Histograma",
                xaxis_title=variable,
                yaxis_title="Frecuencia",
                template="plotly_white"
            )

            # Boxplot
            boxplot_fig = go.Figure(data=[
                go.Box(y=df[variable], marker_color='#004085', boxmean=True)
            ])
            boxplot_fig.update_layout(
                title="Boxplot",
                yaxis_title=variable,
                template="plotly_white"
            )

            # ACF (Autocorrelación)
            acf_values = acf(df[variable].dropna(), nlags=20)
            acf_fig = go.Figure(data=[
                go.Bar(x=list(range(len(acf_values))), y=acf_values, marker_color='#004085')
            ])
            acf_fig.update_layout(
                title="Autocorrelación (ACF)",
                xaxis_title="Lags",
                yaxis_title="ACF",
                template="plotly_white"
            )

            # PACF (Correlación Parcial)
            pacf_values = pacf(df[variable].dropna(), nlags=20)
            pacf_fig = go.Figure(data=[
                go.Bar(x=list(range(len(pacf_values))), y=pacf_values, marker_color='#004085')
            ])
            pacf_fig.update_layout(
                title="Autocorrelación Parcial (PACF)",
                xaxis_title="Lags",
                yaxis_title="PACF",
                template="plotly_white"
            )

            # Diseño para mostrar gráficas
            return html.Div([
                dbc.Row([
                    dbc.Col(dcc.Graph(figure=histogram_fig), width=6),
                    dbc.Col(dcc.Graph(figure=boxplot_fig), width=6),
                ], style={'marginTop': '20px'}),
                html.Div(dcc.Graph(figure=decomposition_fig), style={'marginTop': '20px'}),
                dbc.Row([
                    dbc.Col(dcc.Graph(figure=acf_fig), width=6),
                    dbc.Col(dcc.Graph(figure=pacf_fig), width=6),
                ], style={'marginTop': '20px'})
            ])
        else:
            # Si no hay variable seleccionada o válida
            return html.Div([html.H5('Seleccione una variable válida.', style={'color': 'red'})])

    elif tab == 'tab-bivariado':
        return html.Div([
            # Sección de selección de variables con diseño mejorado
            html.Div([
                html.Div([
                    html.Label('Variable X:', style={
                        'fontWeight': 'bold', 
                        'color': '#004085', 
                        'textAlign': 'center',
                        'marginBottom': '5px',
                        'fontSize': '16px'
                    }),
                    dcc.Dropdown(
                        id='dropdown-x',
                        options=[{'label': col, 'value': col} for col in variables_a_analizar],
                        value=variables_a_analizar[0] if variables_a_analizar else None,
    className="dropdown-custom"
                    )
                ], style={
                    'width': '45%',
                    'marginRight': '5%',
                    'display': 'inline-block',
                    'verticalAlign': 'top'
                }),
                html.Div([
                    html.Label('Variable Y:', style={
                        'fontWeight': 'bold', 
                        'color': '#004085', 
                        'textAlign': 'center',
                        'marginBottom': '5px',
                        'fontSize': '16px'
                    }),
                    dcc.Dropdown(
                        id='dropdown-y',
                        options=[{'label': col, 'value': col} for col in variables_a_analizar],
                        value=variables_a_analizar[1] if len(variables_a_analizar) > 1 else None,
    className="dropdown-custom"
                    )
                ], style={
                    'width': '45%',
                    'display': 'inline-block',
                    'verticalAlign': 'top'
                })
            ], style={
                'display': 'flex',
                'justifyContent': 'space-between',
                'alignItems': 'center',
                'marginBottom': '20px',
                'padding': '10px',
                'backgroundColor': '#f8f9fa',
                'border': '1px solid #004085',
                'borderRadius': '8px'
            }),

            # Gráficos (Placeholder o sección para gráficos)
            html.Div([
                dcc.Graph(id='scatter-plot'),
                dcc.Graph(id='regplot')
            ], style={
                'marginTop': '20px'
            })
        ])

# Nuevo layout para Modelos

# Rutas de los archivos Pickle
# Rutas de los archivos pickle
pickles_metricas = {
    "100m_1": "resultados_altura1.pkl",
    "100m_2": "resultados_altura2.pkl",
    "80m_1": "resultados_altura3.pkl",
    "80m_2": "resultados_altura4.pkl",
    "60m": "resultados_altura5.pkl"
}

pickles_graficos = {
    "100m_1": "resultados_graficos1.pkl",
    "100m_2": "resultados_graficos2.pkl",
    "80m_1": "resultados_graficos3.pkl",
    "80m_2": "resultados_graficos4.pkl",
    "60m": "resultados_graficos5.pkl"
}

pickles_graficos_originales = {
    "100m_1": "resultados_graficos1_original.pkl",
    "100m_2": "resultados_graficos2_original.pkl",
    "80m_1": "resultados_graficos3_original.pkl",
    "80m_2": "resultados_graficos4_original.pkl",
    "60m": "resultados_graficos5_original.pkl"
}

# Dropdown de alturas
alturas = list(pickles_metricas.keys())




alturas = ["100m_1", "100m_2", "80m_1", "80m_2", "60m"]
modelos = []


# Layout completo para la página de Modelos
layout_pagina_modelos = html.Div([
    html.H1("Análisis de Modelos por Altura", style={'textAlign': 'center'}),
    dbc.Row([
        dbc.Col([
            html.Label("Seleccione una altura:"),
            dcc.Dropdown(
                id="dropdown-altura",
                options=[{'label': altura, 'value': altura} for altura in alturas],
                value=alturas[0],  # Altura seleccionada por defecto
                className="dropdown-custom"
            )
        ], width=4),
        dbc.Col([
            html.Label("Seleccione un modelo:"),
            dcc.Dropdown(
                id="dropdown-modelo",
                options=[],  # Se actualizará dinámicamente
                value=None,
                className="dropdown-custom"
            )
        ], width=4),
    ], style={'marginBottom': '20px'}),
    
    # Tabla de métricas
    dash_table.DataTable(
        id='tabla-metricas',
        columns=[],  # Se actualizarán dinámicamente
        data=[],  # Se actualizarán dinámicamente
        style_table={'overflowX': 'auto', 'marginTop': '20px', 'width': '100%'},
        style_cell={
            'textAlign': 'center', 
            'padding': '10px', 
            'fontFamily': 'Arial', 
            'fontSize': '14px'
        },
        style_header={
            'backgroundColor': '#004085', 
            'color': 'white', 
            'fontWeight': 'bold'
        },
        style_data={'backgroundColor': '#f8f9fa', 'color': 'black'}
    ),
    
    dbc.Row([
        dbc.Col(dcc.Graph(id="grafico-train"), width=6),
        dbc.Col(dcc.Graph(id="grafico-test"), width=6),
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id="grafico-histograma-residuos"), width=6),
        dbc.Col(dcc.Graph(id="grafico-autocorrelacion-residuos"), width=6),
    ])
], style={'padding': '20px'})


@app.callback(
    Output("dropdown-modelo", "options"),
    Input("dropdown-altura", "value")
)
def actualizar_modelos(altura_seleccionada):
    if altura_seleccionada:
        # Cargar datos de métricas
        with open(pickles_metricas[altura_seleccionada], "rb") as f:
            metricas = pickle.load(f)
        # Generar lista de modelos sin añadir "Modelo Original"
        modelos = metricas['Modelo'].tolist()
        return [{'label': modelo, 'value': modelo} for modelo in modelos]
    return []




import plotly.graph_objects as go

def crear_graficas_predicciones(nombre_modelo, y_train, y_train_pred, y_test, y_test_pred):
    # Limitar los datos a los primeros 100 puntos
    y_train = y_train[:50]
    y_train_pred = y_train_pred[:50]
    y_test = y_test[:50]
    y_test_pred = y_test_pred[:50]

    # Gráfico para los datos de entrenamiento
    fig_train = go.Figure()
    fig_train.add_trace(go.Scatter(
        y=y_train, mode='lines', name='Train Real',
        line=dict(color='#004085', width=2)  # Azul oscuro
    ))
    fig_train.add_trace(go.Scatter(
        y=y_train_pred, mode='lines', name='Train Predicho',
        line=dict(color='#66B2FF', dash='dot', width=2)  # Azul claro
    ))
    fig_train.update_layout(
        title=f"Resultados de Entrenamiento ({nombre_modelo})",
        xaxis=dict(title="Índice", range=[0, 50]),
        yaxis=dict(
            title="Valores",
            range=[
                min(min(y_train), min(y_train_pred)) - 2,
                max(max(y_train), max(y_train_pred)) + 2
            ]
        ),
        template="plotly_white",
        legend=dict(orientation="h", x=0.5, xanchor="center"),
    )

    # Gráfico para los datos de prueba
    fig_test = go.Figure()
    fig_test.add_trace(go.Scatter(
        y=y_test, mode='lines', name='Test Real',
        line=dict(color='#FF9933', width=2)  # Naranja
    ))
    fig_test.add_trace(go.Scatter(
        y=y_test_pred, mode='lines', name='Test Predicho',
        line=dict(color='#00CC99', dash='dot', width=2)  # Verde
    ))
    fig_test.update_layout(
        title=f"Resultados de Prueba ({nombre_modelo})",
        xaxis=dict(title="Índice", range=[0, 50]),
        yaxis=dict(
            title="Valores",
            range=[
                min(min(y_test), min(y_test_pred)) - 2,
                max(max(y_test), max(y_test_pred)) + 2
            ]
        ),
        template="plotly_white",
        legend=dict(orientation="h", x=0.5, xanchor="center"),
    )

    return fig_train, fig_test



@app.callback(
    [
        Output('tabla-metricas', 'data'),
        Output('tabla-metricas', 'columns'),
        Output('grafico-train', 'figure'),
        Output('grafico-test', 'figure'),
        Output('grafico-histograma-residuos', 'figure'),
        Output('grafico-autocorrelacion-residuos', 'figure')
    ],
    [
        Input("dropdown-altura", "value"),
        Input("dropdown-modelo", "value")
    ]
)
def actualizar_tabla_y_graficos(altura_seleccionada, modelo_seleccionado):
    if altura_seleccionada and modelo_seleccionado:
        modelos_originales = ["WOA - XGBoost", "GWO - XGBoost", "BO - XGBoost"]

        print(f"Altura seleccionada: {altura_seleccionada}")
        print(f"Modelo seleccionado: {modelo_seleccionado}")

        try:
            if modelo_seleccionado in modelos_originales:
                # Buscar en los gráficos originales
                print("Buscando en pickles_graficos_originales...")
                with open(pickles_graficos_originales[altura_seleccionada], "rb") as f:
                    graficos = pickle.load(f)
                print("Contenido de graficos_originales:", [g["nombre_modelo"] for g in graficos])
                datos_graficos = graficos[modelos_originales.index(modelo_seleccionado)]
                print(f"Modelo encontrado en gráficos originales: {modelo_seleccionado}")
            else:
                # Buscar en los gráficos normales
                print("Buscando en pickles_graficos...")
                with open(pickles_graficos[altura_seleccionada], "rb") as f:
                    graficos = pickle.load(f)
                print("Contenido de graficos normales:", [g["nombre_modelo"] for g in graficos])

                # Generar el índice dinámico basado en nombre_modelo
                modelos_indices = {g["nombre_modelo"]: idx for idx, g in enumerate(graficos)}
                print("Índices de modelos:", modelos_indices)

                if modelo_seleccionado not in modelos_indices:
                    raise ValueError(f"El modelo '{modelo_seleccionado}' no está en los gráficos.")
                datos_graficos = graficos[modelos_indices[modelo_seleccionado]]

            # Crear gráficos
            fig_train = go.Figure()
            fig_train.add_trace(go.Scatter(
                y=datos_graficos["y_train"][:50], mode="lines", name="Train Real"
            ))
            fig_train.add_trace(go.Scatter(
                y=datos_graficos["y_train_pred"][:50], mode="lines", name="Train Predicho"
            ))
            fig_train.update_layout(title="Train")

            fig_test = go.Figure()
            fig_test.add_trace(go.Scatter(
                y=datos_graficos["y_test"][:50], mode="lines", name="Test Real"
            ))
            fig_test.add_trace(go.Scatter(
                y=datos_graficos["y_test_pred"][:50], mode="lines", name="Test Predicho"
            ))
            fig_test.update_layout(title="Test")

            fig_histograma = go.Figure()
            fig_histograma.add_trace(go.Histogram(x=datos_graficos["residuos"]))
            fig_histograma.update_layout(title="Histograma de Residuos")

            acf_values = acf(datos_graficos["residuos"], nlags=20)
            fig_autocorrelacion = go.Figure()
            fig_autocorrelacion.add_trace(go.Scatter(
                x=list(range(len(acf_values))),
                y=acf_values,
                mode="lines+markers",
                name="Autocorrelación"
            ))
            fig_autocorrelacion.update_layout(title="Autocorrelación de Residuos")

            # Obtener datos de métricas
            print("Buscando en pickles_metricas...")
            with open(pickles_metricas[altura_seleccionada], "rb") as f:
                metricas = pickle.load(f)
            print("Contenido de métricas:", metricas)

            if modelo_seleccionado in modelos_originales:
                modelo_datos = metricas.iloc[modelos_originales.index(modelo_seleccionado)].to_dict()
            else:
                modelo_datos = metricas.loc[metricas['Modelo'] == modelo_seleccionado].to_dict('records')[0]

            columnas = [{'name': key, 'id': key} for key in modelo_datos.keys()]
            datos = [modelo_datos]

            return datos, columnas, fig_train, fig_test, fig_histograma, fig_autocorrelacion

        except Exception as e:
            print(f"Error procesando el modelo seleccionado: {e}")
            raise e

    # Caso donde no se selecciona nada
    return [], [], go.Figure(), go.Figure(), go.Figure(), go.Figure()


# Actualización del layout principal
app.layout = html.Div([
    dbc.Row([
        # Sidebar
        dbc.Col([
            html.Div([
                html.Img(src='/assets/images.png', style={
                    'width': '120px',
                    'display': 'block',
                    'margin': '0 auto',
                    'marginTop': '20px'
                }),
                html.Hr(style={'border': '2px solid #004085'}),
                dbc.Nav([
                    dbc.NavLink('Inicio', href='/', active='exact', className='nav-link-custom'),
                    dbc.NavLink('EDA', href='/eda', active='exact', className='nav-link-custom'),
                    dbc.NavLink('Modelos', href='/modelos', active='exact', className='nav-link-custom'),
                ], vertical=True, pills=True),
            ], style={
                'position': 'fixed', 
                'top': 0, 
                'left': 0, 
                'bottom': 0,
                'width': '12rem', 
                'backgroundColor': '#004085', 
                'color': 'white',
                'padding': '20px', 
                'boxSizing': 'border-box', 
                'textAlign': 'center'
            })
        ], width=2, style={'padding': '10px'}),  # Asegúrate de que el padding sea 0

        # Main Content
        dbc.Col([
            dcc.Location(id='url', refresh=False),
            html.Div(id='page-content', style={
                'padding': '20px',
                'fontSize': '16px',
                'boxSizing': 'border-box',
                'backgroundColor': '#f8f9fa'
            }),
        ], width=10, style={'padding': '10px'})  # Ajuste de padding en la columna principal
    ], style={'margin': '0', 'padding': '0', 'width': '100%'})  # Elimina márgenes y paddings del dbc.Row
])

# Callback para manejar el contenido dinámico en la barra lateral
@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')]
)
def display_page(pathname):
    if pathname in ['/', '/inicio']:  # Agregar ambas rutas
        return layout_pagina_inicio
    elif pathname == '/eda':
        return layout_pagina_eda
    elif pathname == '/modelos':  # Nuevo apartado
        return layout_pagina_modelos
    else:
        return html.Div([
            html.H2('404: Página no encontrada', style={'color': 'red', 'textAlign': 'center'})
        ])


if __name__ == '__main__':
    app.run_server(debug=True)

