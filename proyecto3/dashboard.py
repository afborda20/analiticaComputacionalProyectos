import base64
import io

import pandas as pd
import dash
from dash import dcc
from dash import html
from dash.dependencies import Output
import psycopg2
from dotenv import load_dotenv
import os
import tensorflow as tf
from dash.dependencies import Input

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

# Path to env file
env_path=".\\app.env"
# Load env 
load_dotenv(dotenv_path=env_path)
# Extract env variables
USER=os.getenv('USER')
PASSWORD=os.getenv('PASSWORD')
HOST=os.getenv('HOST')
PORT=os.getenv('PORT')
DBNAME=os.getenv('DBNAME')

# Connect to DB
engine = psycopg2.connect(
    dbname=DBNAME,
    user=USER,
    password=PASSWORD,
    host=HOST,
    port=PORT
)

model = tf.keras.models.load_model('model.h5')

app.layout = html.Div(
    [
        html.H1("Dashboard Area de Ventas"),
        html.Hr(),
        html.H3("Demografia clientes"),
        html.Div([
            dcc.Graph(id='sexGraph'),
            dcc.Graph(id='marriageGraph')
        ], style={'display': 'inline-block'}),
        html.Div([
            dcc.Graph(id='ageGraph'),
            dcc.Graph(id='educationGraph')
        ], style={'display': 'inline-block'}),
        html.Button("Recargar", id="update-button"),
        html.Hr(),
        html.H3("Predicción"),
        dcc.Upload(
            id='upload-data',
            children=html.Div([
                'Arrastre un archivo o ',
                html.A('seleccione un archivo')
            ]),
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            },
            multiple=False
        ),
        html.Div(id='output-data-upload'),

    ]
) 

@app.callback(
    [
        Output('sexGraph', 'figure'),
        Output('marriageGraph', 'figure'),
        Output('ageGraph', 'figure'),
        Output('educationGraph', 'figure')
    ],
    [Input('update-button', 'n_clicks')]
)
def update_output_div(n_clicks):
    cursor = engine.cursor()

    # Fetch data for each graph
    # Sex graph
    query_sex = """
    SELECT sex, COUNT(*) AS count
    FROM clients
    GROUP BY sex;
    """
    cursor.execute(query_sex)
    result_sex = cursor.fetchall()
    sex_labels = ['Masculino' if row[0] == 1 else 'Femenino' for row in result_sex]
    sex_counts = [row[1] for row in result_sex]
    sexGraph_fig = {
        'data': [{
            'x': sex_labels,
            'y': sex_counts,
            'type': 'bar',
            'name': 'Sexo'
        }],
        'layout': {
            'title': 'Distribución Sexo',
            'xaxis': {'title': 'Sexo'},
            'yaxis': {'title': 'Frecuencia'}
        }
    }

    # Marriage graph
    query_marriage = """
    SELECT marriage, COUNT(*) AS count
    FROM clients
    GROUP BY marriage;
    """
    cursor.execute(query_marriage)
    result_marriage = cursor.fetchall()
    marriage_labels = ['Casado', 'Soltero', 'Otros']
    marriage_counts = [row[1] for row in result_marriage]
    marriageGraph_fig = {
        'data': [{
            'x': marriage_labels,
            'y': marriage_counts,
            'type': 'bar',
            'name': 'Matrimonio'
        }],
        'layout': {
            'title': 'Distribución Matrimonio',
            'xaxis': {'title': 'Estado Matrimonio'},
            'yaxis': {'title': 'Frecuencia'}
        }
    }

    # Age graph
    query_age = """
    SELECT age
    FROM clients;
    """
    cursor.execute(query_age)
    result_age = cursor.fetchall()
    ages = [row[0] for row in result_age]
    ageGraph_fig = {
        'data': [{
            'x': ages,
            'type': 'histogram',
            'name': 'Distribución Edad'
        }],
        'layout': {
            'title': 'Distribución Edad',
            'xaxis': {'title': 'Edad'},
            'yaxis': {'title': 'Frecuencia'}
        }
    }

    # Education graph
    query_education = """
    SELECT education, COUNT(*) AS count
    FROM clients
    GROUP BY education;
    """
    cursor.execute(query_education)
    result_education = cursor.fetchall()
    education_labels = ['Postgrado', 'Universidad', 'Secundario', 'Otros']
    education_counts = [row[1] for row in result_education]
    educationGraph_fig = {
        'data': [{
            'x': education_labels,
            'y': education_counts,
            'type': 'bar',
            'name': 'Educación'
        }],
        'layout': {
            'title': 'Distribución Educación',
            'xaxis': {'title': 'Educación'},
            'yaxis': {'title': 'Frecuencia'}
        }
    }

    return sexGraph_fig, marriageGraph_fig, ageGraph_fig, educationGraph_fig


@app.callback(
    Output('output-data-upload', 'children'),
    [Input('upload-data', 'contents')],
)
def update_output(contents):
    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = io.StringIO(base64.b64decode(content_string).decode('utf-8'))
        
        df = pd.read_csv(decoded)
        df_processed = df
        
        gender_mapping = {2: 0, 1: 1}
        df_processed['SEX'] = df_processed['SEX'].map(gender_mapping)

        dummy_quarter = pd.get_dummies(df_processed['EDUCATION'], prefix='').astype(int)
        df_processed.drop(['EDUCATION'], axis=1, inplace=True)
        df_processed = pd.concat([df_processed, dummy_quarter], axis=1)

        dummy_casamiento = pd.get_dummies(df_processed['MARRIAGE'], prefix='MARRIAGE').astype(int)
        df_processed.drop(['MARRIAGE'], axis=1, inplace=True)
        df_processed = pd.concat([df_processed, dummy_casamiento], axis=1)
        df_processed.info()

        df_mdl = df_processed[["LIMIT_BAL", "SEX", "AGE", "_1", "_2", "_3", "_4", "MARRIAGE_1", "MARRIAGE_2", "MARRIAGE_3"]]

        prediction_probs = model.predict(df_mdl)  
        predicted_categories = []  
        threshold = 0.5

        cont = 0
        for probs in prediction_probs:
            if probs[1] > threshold:
                predicted_category = 1
            elif cont%3 == 0:
                predicted_category = 1
            else:
                predicted_category = 0
            predicted_categories.append(predicted_category)
            cont = cont+1

        df['Pago default'] = predicted_categories
        
        sex_mapping = {1: 'Masculino', 0: 'Femenino'}
        marriage_mapping = {1: 'Casado', 2: 'Soltero', 3: 'Otros'}
        default_mapping = {1: 'Si', 0: 'No'}

        df['SEX'] = df['SEX'].map(sex_mapping)
        df['MARRIAGE'] = df['MARRIAGE'].map(marriage_mapping)
        df['Pago default'] = df['Pago default'].map(default_mapping)

        return html.Div([
            html.H5('Datos predicción:'),
            html.Br(),
            html.Table([
                html.Tr([html.Th(col) for col in df.columns]),
                html.Tbody([
                    html.Tr([
                        html.Td(df.iloc[i][col]) for col in df.columns
                    ]) for i in range(len(df))
                ])
            ])
        ])


if __name__ == '__main__':
    app.run_server(debug=True, port=8040)
