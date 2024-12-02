import dash
import pandas as pd
from dash import dcc, html
from dash import dash_table
import plotly.graph_objs as go
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import numpy as np
import io
import base64
from statsmodels.tsa.stattools import acf, pacf
# Configurar el tema profesional de Dash Bootstrap Components
app = dash.Dash(
    external_stylesheets=[dbc.themes.MINTY], 
    suppress_callback_exceptions=True
)

# Cargar datos desde el archivo
try:
    file_path = "C:/Users/EMANUEL/Downloads/ws.xlsx"
    df = pd.read_excel(file_path)
except Exception as e:
    df = pd.DataFrame()  # En caso de error, usar un DataFrame vacío
    print(f"Error al cargar los datos: {e}")

# Verificar si 'Fecha' existe y convertirla a formato datetime
if not df.empty and 'Fecha' in df.columns:
    df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
    df['Mes'] = df['Fecha'].dt.month

# Calcular variables para el análisis
variables_a_analizar = [col for col in df.columns if df[col].dtype in ['float64', 'int64']]
missing_data = df.isnull().mean() * 100 if not df.empty else pd.Series()

# Layout de inicio
layout_pagina_inicio = html.Div([
    html.H1('Bienvenido al Dashboard de Análisis', style={
        'textAlign': 'center', 'fontWeight': 'bold', 
        'fontSize': '2.5rem', 'color': '#004085'
    }),
    html.P('Explora los datos y modelos relacionados con energía eólica. Selecciona una pestaña en el menú.', style={
        'textAlign': 'center', 'marginTop': '20px'
    })
])

# Layout de EDA
# Layout de EDA actualizado
layout_pagina_eda = html.Div([
    html.H1('Análisis Exploratorio de Datos', style={
        'fontWeight': 'bold', 'textAlign': 'center', 'color': '#004085'
    }),
    # Gráfica de datos faltantes
    html.H3('Análisis General', style={'marginTop': '20px', 'color': '#004085'}),
    dcc.Graph(figure=go.Figure(data=[
        go.Bar(
            x=missing_data.index if not missing_data.empty else [],
            y=missing_data.values if not missing_data.empty else [],
            name='Faltantes',
            marker_color='#004085'
        )
    ]).update_layout(
        title="Porcentaje de Datos Faltantes por Variable",
        xaxis_title="Variables",
        yaxis=dict(title="Porcentaje", range=[0, 100]),
        template='plotly_white'
    )),
    # Cuadritos informativos
    dbc.Row([
        dbc.Col(html.Div([
            html.Div("Número de Observaciones", className="info-title"),
            html.Div(f"{df.shape[0]}" if not df.empty else "0", className="info-value")
        ], className="info-box"), width=6),
        dbc.Col(html.Div([
            html.Div("Número de Variables", className="info-title"),
            html.Div(f"{df.shape[1]}" if not df.empty else "0", className="info-value")
        ], className="info-box"), width=6),
    ], style={'marginTop': '20px'}),

    # Diccionario de datos
    html.H3('Diccionario', style={'marginTop': '20px', 'color': '#004085'}),
    html.Div([
        dash_table.DataTable(
            id='tabla-tipos',
            columns=[{'name': 'Variable', 'id': 'Variable'}, {'name': 'Tipo', 'id': 'Tipo'}],
            data=[{'Variable': col, 'Tipo': str(df[col].dtype)} for col in df.columns] if not df.empty else [],
            style_table={'overflowX': 'auto'},
            style_cell={
                'textAlign': 'center', 
                'padding': '10px',
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
    ], style={'marginTop': '20px'}),
    # Dropdown de selección de variable (inicialmente vacío)
    html.Div([
        html.Label('Selecciona una variable:', style={'fontWeight': 'bold', 'color': '#004085'}),
        dcc.Dropdown(
            id='variable-dropdown',
            options=[{'label': var, 'value': var} for var in variables_a_analizar],
            value=variables_a_analizar[0] if variables_a_analizar else None,
            style={'width': '50%'}
        )
    ], style={'marginTop': '20px'}),
    # Pestañas de análisis
    dcc.Tabs(id='tabs-eda', value='tab-univariado', className='custom-tabs', children=[
        dcc.Tab(label='Univariado', value='tab-univariado', className='custom-tab', selected_className='custom-tab--selected'),
        dcc.Tab(label='Bivariado', value='tab-bivariado', className='custom-tab', selected_className='custom-tab--selected'),
    ], style={'marginTop': '20px'}),
    html.Div(id='tabs-eda-content', style={'marginTop': '20px'})
])



# Layout de Modelos
layout_pagina_modelos = html.Div([
    html.H1('Modelos Predictivos', style={
        'fontWeight': 'bold', 'textAlign': 'center', 'color': '#004085'
    }),
    dcc.Tabs(id='tabs-modelos', value='tab-modelo1', children=[
        dcc.Tab(label='Modelo 1', value='tab-modelo1', style={'fontWeight': 'bold'}),
        dcc.Tab(label='Modelo 2', value='tab-modelo2', style={'fontWeight': 'bold'}),
        dcc.Tab(label='Modelo 3', value='tab-modelo3', style={'fontWeight': 'bold'}),
    ]),
    html.Div(id='tabs-modelos-content', style={'marginTop': '20px'})
])

# Layout de la app
app.layout = html.Div([
    dbc.Row([
        dbc.Col([
            html.Div([
                html.Div([html.Img(src='/assets/images.png', style={'width': '150px', 'marginBottom': '20px'})]),
                html.Hr(style={'border': '2px solid #004085'}),
                dbc.Nav([
                    dbc.NavLink('Inicio', href='/', active='exact', className='nav-link-custom'),
                    dbc.NavLink('EDA', href='/eda', active='exact', className='nav-link-custom'),
                    dbc.NavLink('Modelos', href='/modelos', active='exact', className='nav-link-custom'),
                ], vertical=True, pills=True),
            ], style={
                'position': 'fixed', 'top': 0, 'left': 0, 'bottom': 0, 
                'width': '12rem', 
                'padding': '15px', 
                'box-sizing': 'border-box', 
                'backgroundColor': '#004085', 'color': '#FFFFFF'
            })
        ], width=2),
        dbc.Col([
            dcc.Location(id='url', refresh=False),
            html.Div(id='page-content', style={
                'padding': '20px',
                'margin-left': '4rem', 
                'box-sizing': 'border-box',
            })
        ], width=10)
    ])
])

# CSS adicional
# CSS adicional para mejorar el estilo de las pestañas
app.index_string += '''
<style>
/* Estilo personalizado para las pestañas */
.custom-tabs {
    font-family: Arial, sans-serif;
    background-color: #004085;
    border-radius: 8px;
    padding: 10px;
}

.custom-tabs {
    font-family: Arial, sans-serif;
    background-color: #004085;
    border-radius: 8px;
    padding: 10px;
}

.custom-tab {
    color: grey; /* Color del texto en las pestañas inactivas */
    font-weight: bold;
    padding: 10px;
    border-radius: 5px;
    margin-right: 5px;
    transition: background-color 0.3s ease, color 0.3s ease;
    text-align: center;
}

.custom-tab:hover {
    background-color: #0062cc;
    color: #004085; /* Color del texto cuando pasas el ratón */
}

.custom-tab--selected {
    background-color: white;
    color: #004085; /* Color del texto en la pestaña activa */
    font-weight: bold;
    border: 2px solid #004085;
    text-align: center;
}

.info-box {
    background-color: #004085;
    color: white;
    border-radius: 10px;
    padding: 20px;
    text-align: center;
    font-weight: bold;
}
.info-title {
    font-size: 16px;
    margin-bottom: 10px;
}
.info-value {
    font-size: 24px;
}


</style>
'''

if not df.empty and 'Fecha' in df.columns:
    df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
    df = df.set_index('Fecha')

variables_a_analizar = [col for col in df.columns if df[col].dtype in ['float64', 'int64']]
# Callback para manejar las páginas
@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')]
)
def display_page(pathname):
    if pathname == '/':
        return layout_pagina_inicio
    elif pathname == '/eda':
        return layout_pagina_eda
    elif pathname == '/modelos':
        return layout_pagina_modelos
    else:
        return html.Div([
            html.H2('404: Página no encontrada', style={'textAlign': 'center', 'color': 'red'})
        ])

# Callback para las pestañas del EDA
@app.callback(
    Output('tabs-eda-content', 'children'),
    [Input('tabs-eda', 'value'),
     Input('variable-dropdown', 'value')]
)
def render_eda_tabs(tab, variable):
    if tab == 'tab-univariado':
        if variable and variable in df.columns:
            # Descomposición de la serie
            decomposition = seasonal_decompose(df[variable], model='additive', period=12)
            observed = decomposition.observed
            trend = decomposition.trend
            seasonal = decomposition.seasonal
            residual = decomposition.resid

            # Gráficos de autocorrelación
            acf_fig = plot_autocorrelation(df[variable])
            pacf_fig = plot_partial_autocorrelation(df[variable])

            # Histograma y boxplot
            histogram_fig = go.Figure(data=[
                go.Histogram(x=df[variable], nbinsx=30, marker_color='#004085')
            ]).update_layout(title="Histograma", xaxis_title=variable, yaxis_title="Frecuencia")
            boxplot_fig = go.Figure(data=[
                go.Box(y=df[variable], marker_color='#004085', boxmean=True)
            ]).update_layout(title="Boxplot", yaxis_title=variable)

            # Gráfico de la serie
            series_fig = go.Figure(data=[
                go.Scatter(x=df.index, y=df[variable], mode='lines', line=dict(color='#004085'))
            ]).update_layout(title="Serie Temporal", xaxis_title="Fecha", yaxis_title=variable)

            # Gráficos de descomposición
# Gráficos de descomposición
            decomposition_fig = go.Figure()
            decomposition_fig.add_trace(go.Scatter(
                x=df.index, y=observed, mode='lines', name='Observed', visible='legendonly'))
            decomposition_fig.add_trace(go.Scatter(
                x=df.index, y=trend, mode='lines', name='Trend', visible='legendonly'))
            decomposition_fig.add_trace(go.Scatter(
                x=df.index, y=seasonal, mode='lines', name='Seasonal', visible='legendonly'))
            decomposition_fig.add_trace(go.Scatter(
                x=df.index, y=residual, mode='lines', name='Residual', visible='legendonly'))
            decomposition_fig.update_layout(
                title="Descomposición de la Serie", 
                xaxis_title="Fecha", 
                yaxis_title=variable,
                template="plotly_white"  # Opcional: Mejora estética
            )


            return html.Div([
                # Dropdown para seleccionar la variable
                html.Div([
                    html.Label('Selecciona una variable:', style={'fontWeight': 'bold', 'color': '#004085'}),
                    dcc.Dropdown(
                        id='variable-dropdown',
                        options=[{'label': var, 'value': var} for var in variables_a_analizar],
                        value=variable,
                        style={'width': '50%'}
                    )
                ], style={'marginBottom': '20px'}),

                    html.Div([
                        html.H5(f'Resumen Estadístico de {variable}', style={'color': '#004085'}),
                        dash_table.DataTable(
                            id='resumen-estadistico',
                            columns=[{'name': col, 'id': col} for col in ['Estadística'] + list(df[variable].describe().index)],
                            data=[{'Estadística': 'Valor'} | df[variable].describe().round(2).to_dict()],
                            style_table={'overflowX': 'auto'},
                            style_cell={
                                'textAlign': 'center',
                                'padding': '10px',
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
                    ], style={'marginTop': '20px'}),


                # Gráficos integrados
                html.Div([
                    html.H5('Gráficos Integrados', style={'color': '#004085', 'textAlign': 'center'}),
                    dbc.Row([
                        dbc.Col(dcc.Graph(figure=series_fig), width=6),
                        dbc.Col(dcc.Graph(figure=decomposition_fig), width=6)
                    ]),
                    dbc.Row([
                        dbc.Col(dcc.Graph(figure=acf_fig), width=6),
                        dbc.Col(dcc.Graph(figure=pacf_fig), width=6)
                    ]),
                    dbc.Row([
                        dbc.Col(dcc.Graph(figure=histogram_fig), width=6),
                        dbc.Col(dcc.Graph(figure=boxplot_fig), width=6)
                    ])
                ])
            ])
        else:
            return html.Div([
                html.H5('Seleccione una variable válida para el análisis.', style={'color': 'red', 'textAlign': 'center'})
            ])
    elif tab == 'tab-bivariado':
        return html.Div([
            html.H5('Análisis Bivariado en construcción...', style={'color': '#004085', 'textAlign': 'center'})
        ])


# Función para crear la gráfica de autocorrelación
def plot_autocorrelation(series):
    acf_values = acf(series.dropna(), nlags=20)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(range(len(acf_values))), y=acf_values, marker_color='#004085'))
    fig.update_layout(title="Autocorrelación", xaxis_title="Lags", yaxis_title="ACF")
    return fig

# Función para crear la gráfica de correlación parcial
def plot_partial_autocorrelation(series):
    pacf_values = pacf(series.dropna(), nlags=20)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(range(len(pacf_values))), y=pacf_values, marker_color='#004085'))
    fig.update_layout(title="Correlación Parcial", xaxis_title="Lags", yaxis_title="PACF")
    return fig

def fig_to_plotly(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)
    encoded_image = base64.b64encode(buffer.read()).decode('utf-8')
    buffer.close()
    return dcc.Graph(figure={}, config={'displayModeBar': False}, id='acf-pacf-plot', src='data:image/png;base64,{}'.format(encoded_image))
# Ejecución de la app
if __name__ == '__main__':
    app.run_server(debug=True)
