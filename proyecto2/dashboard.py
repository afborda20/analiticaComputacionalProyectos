import dash
from dash import dcc
from dash import html
from dash.dependencies import Output
import psycopg2
from dotenv import load_dotenv
import os
import tensorflow as tf

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

model = tf.keras.models.load_model('model.keras')

app.layout = html.Div(
    [
        html.H1("Demografia Clientes"),
        dcc.Graph(id='sexGraph'),
        dcc.Graph(id='marriageGraph'),
        dcc.Graph(id='ageGraph'),
        dcc.Graph(id='educationGraph')
    ]
)

@app.callback(
    [
        Output('sexGraph', 'figure'),
        Output('marriageGraph', 'figure'),
        Output('ageGraph', 'figure'),
        Output('educationGraph', 'figure')
    ],
    []
)
def update_output_div():
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
            'yaxis': {'title': 'Conteo'}
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
            'yaxis': {'title': 'Cuenta'}
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
            'yaxis': {'title': 'Frequency'}
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
            'yaxis': {'title': 'Cuenta'}
        }
    }

    return sexGraph_fig, marriageGraph_fig, ageGraph_fig, educationGraph_fig

if __name__ == '__main__':
    app.run_server(debug=True, port=8040)
