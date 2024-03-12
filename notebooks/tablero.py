import dash
from dash import dcc, html, Input, Output
import pandas as pd
import io
import base64
import plotly.express as px
import statsmodels.api as sm
from dash.dependencies import Input, Output, State
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import cross_val_score
import statsmodels.api as sm

# Crear la aplicación Dash
app = dash.Dash(__name__, external_stylesheets=['/assets/styles.css'])

# Definir el diseño de la aplicación
app.layout = html.Div([
    html.H1("ACTD Proyecto - Regresión", id="title"),
    dcc.Upload(
        id='upload-data',
        children=html.Button('Cargar archivo CSV'),
        multiple=False
    ),
    html.Div(id='output-data-upload'),
    html.Div([
        html.Label('Seleccione las variables independientes:'),
        dcc.Dropdown(
            id='dropdown-variables',
            options=[],
            multi=True
        ),
        html.Label('Variable de interés:'),
        dcc.Dropdown(
            id='dropdown-interest',
            options=[],
        ),
    ]),
    html.Button('Actualizar', id='button-update'),
    html.Div(id='output-table'),
    html.Div([
        html.Div(id='selected-columns', className='column'),  # Mantenemos la tabla del modelo en una columna
        html.Div(id='model-info-table', className='column'),  # Agregamos la tabla de indicadores en otra columna
    ], className='row'),  # Envuelva las columnas en una fila
])


def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string).decode('utf-8')
    try:
        if 'csv' in filename:
            # Se lee el archivo CSV y se carga en un DataFrame
            df = pd.read_csv(io.StringIO(decoded))
            return df
        else:
            return None
    except Exception as e:
        print(e)
        return None

@app.callback(Output('output-data-upload', 'children'),
              [Input('upload-data', 'contents')],
              [Input('upload-data', 'filename')])
def update_output(contents, filename):
    if contents is not None:
        return html.H5(f'Archivo cargado: {filename}')

@app.callback(
    [Output('dropdown-variables', 'options'),
     Output('dropdown-interest', 'options')],
    [Input('upload-data', 'contents'),
     Input('upload-data', 'filename')])
def update_dropdown_options(contents, filename):
    if contents is not None:
        df = parse_contents(contents, filename)
        if df is not None:
            options = [{'label': col, 'value': col} for col in df.columns]
            return options, options
    return [], []

@app.callback(
    Output('output-table', 'children'),
    [Input('upload-data', 'contents'),
     Input('upload-data', 'filename'),
     Input('dropdown-variables', 'value'),
     Input('dropdown-interest', 'value')])
def update_table(contents, filename, selected_variables, interest_variable):
    if contents is not None and selected_variables and interest_variable:
        # Leer el archivo CSV y cargarlo en un DataFrame
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string).decode('utf-8')
        df = pd.read_csv(io.StringIO(decoded))
        
        # Filtrar el DataFrame con las columnas seleccionadas
        selected_df = df[selected_variables]
        
        # Seleccionar solo las primeras 10 filas
        selected_df = selected_df.head(10)
        
        # Crear una lista de filas para la tabla HTML
        table_rows = []
        for i in range(len(selected_df)):
            table_row = []
            for col in selected_variables:
                value = selected_df.iloc[i][col]
                table_row.append(html.Td(value))
            table_rows.append(html.Tr(table_row))
        
        # Crear la tabla HTML
        table = html.Table([
            html.Thead(html.Tr([html.Th(col) for col in selected_variables])),
            html.Tbody(table_rows)
        ])
        return table
    return None

@app.callback(
    Output('selected-columns', 'children'),
    [Input('button-update', 'n_clicks')],
    [State('upload-data', 'contents'),
     State('upload-data', 'filename'),
     State('dropdown-variables', 'value'),
     State('dropdown-interest', 'value'),
     State('selected-columns', 'children')])
def update_selected_columns(n_clicks, contents, filename, selected_variables, interest_variable, selected_columns_text):
    if n_clicks is not None and selected_variables and interest_variable:
        # Obtener el DataFrame con la información de las columnas seleccionadas
        df = parse_contents(contents, filename)
        
        # Verificar si se ha cargado el DataFrame correctamente
        if df is not None:
            # Realizar el procesamiento adicional necesario (por ejemplo, crear un modelo)
            model,corte, coef, loss, anova, y_test, y_pred = crearmodelo(df, selected_variables, interest_variable)
            
            # Organizar los resultados del análisis de varianza (ANOVA) en formato legible
            anova_text = organizar_resultados_anova(anova)
            column0=["Rsquared","Rsquared-Adjusted","F-Value","Punto de corte","MAE","MSE","RMSE"]
            column1=[model.rsquared,model.rsquared_adj,model.fvalue,corte,loss[0],loss[1],loss[2]]
            # Crear la tabla con los coeficientes
            table_header = html.Tr([html.Th('Indicadores'), html.Th('Valor')])
            table_rows = [
                html.Tr([html.Td(column0[i]), html.Td(column1[i])]) for i in range(len(column0))
            ]
            coef_table = html.Table([table_header] + table_rows)
            
            # Actualizar el texto de las columnas seleccionadas con los resultados
            selected_columns_text = (
                f'Punto de corte: {corte}\n'
                f'Funciones de Pérdida del modelo: MAE: {loss[0]}, MSE :{loss[1]}, RMSE :{loss[2]} \n'
                f'Resultados del análisis de varianza (ANOVA):\n{anova_text}'
            )
            
            # Generar gráficos con Plotly Express
            fig1 = px.scatter(x=y_test, y=y_pred, labels={'x': 'Valores reales', 'y': 'Prediccion del modelo'}, title='Regresion')
            fig1.add_scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], mode='lines', line=dict(color='red', dash='dash'), name='Línea de referencia y=x')
            # Aquí deberías generar el gráfico para 'influence.png' usando Plotly Express si es posible.
            
            return [html.P(selected_columns_text), coef_table, dcc.Graph(figure=fig1)]  # Agregar la tabla de coeficientes al retorno
    return None

@app.callback(
    Output('model-info-table', 'children'),  # Nueva salida para la tabla del modelo
    [Input('button-update', 'n_clicks')],
    [State('upload-data', 'contents'),
     State('upload-data', 'filename'),
     State('dropdown-variables', 'value'),
     State('dropdown-interest', 'value'),
     State('selected-columns', 'children')])
def update_model_info_table(n_clicks, contents, filename, selected_variables, interest_variable, selected_columns_text):
    if n_clicks is not None and selected_variables and interest_variable:
        # Obtener el DataFrame con la información de las columnas seleccionadas
        df = parse_contents(contents, filename)
        
        # Verificar si se ha cargado el DataFrame correctamente
        if df is not None:
            # Realizar el procesamiento adicional necesario (por ejemplo, crear un modelo)
            model,corte, coef, loss, anova, y_test, y_pred= crearmodelo(df, selected_variables, interest_variable)  # Ignoramos otros valores de salida no necesarios
            
            # Crear la tabla con la información del modelo
            model_table_header = html.Tr([html.Th('Variable'), html.Th('Coeficiente'), html.Th('T-value'), html.Th('Standard Error'), html.Th('P-value [0.025      0.975]')])
            model_table_rows = [
                html.Tr([
                    html.Td(row[0]),
                    html.Td(row[1]),
                    html.Td(row[2]),
                    html.Td(row[3]),
                    html.Td(row[4])
                ]) for row in zip(selected_variables, model.params, model.tvalues, model.bse, model.pvalues)
            ]
            model_table = html.Table([model_table_header] + model_table_rows)
            
            return model_table
    return None

def organizar_resultados_anova(anova):
    # Acceder a las tablas de resultados dentro del objeto Summary
    tables = anova.tables

    # La tabla que contiene los coeficientes es la segunda tabla
    coeficientes_table = tables[1]

    # Convertir la tabla de coeficientes en un DataFrame de pandas
    df_coeficientes = pd.DataFrame(coeficientes_table.data[1:], columns=coeficientes_table.data[0])

    return df_coeficientes

def crearmodelo(df, features, resp):
    features = features
    x = df[features]
    x = x.reset_index(drop=True)
    y = df[resp]
    y = y.reset_index(drop=True)
    X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=1, test_size=0.2)
    linreg = LinearRegression()
    linreg.fit(X_train, y_train)
    corte = linreg.intercept_
    coef = list(zip(features, linreg.coef_))
    y_pred = linreg.predict(X_test)
    # plt.scatter(y_test, y_pred, color='blue')  # Esto ya no es necesario
    # plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], linestyle='--', color='red')  # Esto tampoco
    # plt.xlabel('Valores reales')
    # plt.ylabel('Prediccion del modelo')
    # plt.title('Regresion')
    # plt.savefig("regresion.png")
    # plt.close()
    # Ahora métricas
    MAE = metrics.mean_absolute_error(y_test, y_pred)
    MSE = metrics.mean_squared_error(y_test, y_pred)
    RMSE = np.sqrt(MSE)
    loss = [MAE, MSE, RMSE]
    scores = cross_val_score(linreg, x, y, cv=5, scoring='neg_mean_squared_error')
    mse_scores = - scores
    rmse_scores = np.sqrt(mse_scores)
    # Ahora calculamos anova
    X_train = sm.add_constant(X_train)
    model = sm.OLS(y_train, X_train).fit()
    anova = model.summary()
    # fig = sm.graphics.influence_plot(model, criterion="cooks")  # Ya no es necesario
    # fig.savefig('influence.png')
    # plt.close(fig)
    return model,corte, coef, loss, anova, y_test, y_pred

if __name__ == '__main__':
    app.run_server(debug=True)
