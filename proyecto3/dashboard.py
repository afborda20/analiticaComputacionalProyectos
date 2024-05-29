import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

# Load the data
df = pd.read_csv('data/data_limpia.csv')

# Layout of the app
app.layout = html.Div(
    [
        html.H1("Pruebas Saber 2019"),
        html.Hr(),
        html.H3("Distribución del Puntaje Global"),
        dcc.Graph(id='puntGlobalGraph'),
        html.Button("Recargar", id="update-button"),
        html.Hr(),
        html.H3("Datos Demográficos"),
        html.Div([
            dcc.Graph(id='generoGraph', style={'display': 'inline-block', 'width': '25%'}),
            dcc.Graph(id='computadorGraph', style={'display': 'inline-block', 'width': '25%'}),
            dcc.Graph(id='internetGraph', style={'display': 'inline-block', 'width': '25%'}),
            dcc.Graph(id='estratoGraph', style={'display': 'inline-block', 'width': '25%'})
        ]),
        dcc.Markdown('''
        ### Interpretación del modelo
    
        En conclusión, con el MSE más bajo fue con el optimizador de RMSprop con respecto a la variable del puntaje global donde los datos demográficos de los estudiantes y de sus familias son los que causan el menor error deseado en los puntajes en la prueba ICFES. Se recomienda hacer experimentos tomando en cuenta diferentes supuestos relacionados con los datos demográficos para esperar un incremento en la variable del puntaje global y recomendarles los cambios a las entidades nacionales pertinentes.
        ''')
    ]
)

# Callback to update the graphs
@app.callback(
    [
        Output('puntGlobalGraph', 'figure'),
        Output('generoGraph', 'figure'),
        Output('computadorGraph', 'figure'),
        Output('internetGraph', 'figure'),
        Output('estratoGraph', 'figure')
    ],
    [Input('update-button', 'n_clicks')]
)
def update_graphs(n_clicks):
    # Puntaje Global Graph
    punt_global_counts = df['punt_global'].value_counts().sort_index()
    puntGlobal_fig = {
        'data': [{
            'x': punt_global_counts.index,
            'y': punt_global_counts.values,
            'type': 'bar',
            'name': 'Puntaje Global'
        }],
        'layout': {
            'title': 'Distribución del Puntaje Global',
            'xaxis': {'title': 'Puntaje Global'},
            'yaxis': {'title': 'Frecuencia'}
        }
    }

    # Género Graph
    genero_counts = df['estu_genero'].value_counts()
    genero_labels = ['Masculino' if i == 1 else 'Femenino' for i in genero_counts.index]
    genero_fig = {
        'data': [{
            'x': genero_labels,
            'y': genero_counts.values,
            'type': 'bar',
            'name': 'Género'
        }],
        'layout': {
            'title': 'Distribución por Género',
            'xaxis': {'title': 'Género'},
            'yaxis': {'title': 'Frecuencia'}
        }
    }

    # Tiene Computador Graph
    computador_counts = df['fami_tienecomputador'].value_counts()
    computador_labels = ['No', 'Sí']
    computador_fig = {
        'data': [{
            'x': computador_labels,
            'y': computador_counts.values,
            'type': 'bar',
            'name': 'Tiene Computador'
        }],
        'layout': {
            'title': 'Disponibilidad de Computador',
            'xaxis': {'title': 'Tiene Computador'},
            'yaxis': {'title': 'Frecuencia'}
        }
    }

    # Tiene Internet Graph
    internet_counts = df['fami_tieneinternet'].value_counts()
    internet_labels = ['No', 'Sí']
    internet_fig = {
        'data': [{
            'x': internet_labels,
            'y': internet_counts.values,
            'type': 'bar',
            'name': 'Tiene Internet'
        }],
        'layout': {
            'title': 'Disponibilidad de Internet',
            'xaxis': {'title': 'Tiene Internet'},
            'yaxis': {'title': 'Frecuencia'}
        }
    }

    # Estrato Graph
    estrato_counts = df['fami_estratovivienda'].value_counts().sort_index()
    estrato_fig = {
        'data': [{
            'x': estrato_counts.index,
            'y': estrato_counts.values,
            'type': 'bar',
            'name': 'Estrato'
        }],
        'layout': {
            'title': 'Distribución por Estrato',
            'xaxis': {'title': 'Estrato'},
            'yaxis': {'title': 'Frecuencia'}
        }
    }

    return puntGlobal_fig, genero_fig, computador_fig, internet_fig, estrato_fig

if __name__ == '__main__':
    app.run_server(debug=True, port=8040)
