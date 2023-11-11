import streamlit as st
import tkinter as tk
from tkinter import ttk, filedialog
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import openpyxl 
import re
import plotly.express as px
from scipy.stats import chi2_contingency
import numpy as np



def cargar_archivo():
    st.sidebar.title("Menú")
    archivo = st.sidebar.file_uploader("Cargar archivo .csv o .xlsx", type=["csv", "xlsx"])
    if archivo is not None:
        st.sidebar.success("Archivo cargado correctamente!")
        # Guardar el archivo cargado en la sesión de Streamlit
        st.session_state.archivo_cargado = archivo


def clasificar_variables(datos):
    variables_categoricas = list(datos.select_dtypes(include=['object']).columns)

    # Identificar columnas que parecen fechas usando expresiones regulares
    patron_fecha = re.compile(r'\b(?:\d{1,4}[-/]\d{1,2}[-/]\d{1,4}|\d{1,2}[-/]\d{1,2}[-/]\d{1,4}|\d{1,4}[-/]\d{1,2}[-/]\d{1,2})\b')
    posibles_variables_fecha = [col for col in datos.columns if any(patron_fecha.match(str(dato)) for dato in datos[col])]

    # Verificar si las posibles variables de fecha son realmente fechas
    variables_fecha = [col for col in posibles_variables_fecha if pd.api.types.is_datetime64_any_dtype(pd.to_datetime(datos[col], errors='coerce'))]

    # Excluir tipo fecha de variables_categoricas
    variables_categoricas = [col for col in variables_categoricas if col not in variables_fecha]

    variables_numericas_continuas = []
    variables_numericas_discretas = []

    for col in datos.columns:
        # Excluir fechas de variables numéricas
        if col not in variables_fecha:
            unique_count = len(datos[col].unique())
            if pd.api.types.is_numeric_dtype(datos[col]):
                # Clasificar como continua si tiene más de 30 valores distintos
                if unique_count > 30:
                    variables_numericas_continuas.append(col)
                else:
                    variables_numericas_discretas.append(col)

    return variables_categoricas, variables_numericas_continuas, variables_numericas_discretas, variables_fecha


def mostrar_grafica_variable_numerica_continua(datos, variable_seleccionada):
    # Calcular estadísticas
    media = datos[variable_seleccionada].mean()
    mediana = datos[variable_seleccionada].median()
    desviacion_estandar = datos[variable_seleccionada].std()
    varianza = datos[variable_seleccionada].var()
    moda = datos[variable_seleccionada].mode()[0]

    # Mostrar gráfica de densidad con histograma
    fig, ax = plt.subplots()
    sns.histplot(data=datos, x=variable_seleccionada, kde=False, ax=ax)
    ax.axvline(media, color='red', label=f'Media: {media:.2f}')
    ax.axvline(mediana, color='green', label=f'Mediana: {mediana:.2f}')
    ax.legend()
    st.pyplot(fig)


    # Mostrar estadísticas
    st.write(f"**Media:** {media:.2f}")
    st.write(f"**Mediana:** {mediana:.2f}")
    st.write(f"**Desviación Estándar:** {desviacion_estandar:.2f}")
    st.write(f"**Varianza:** {varianza:.2f}")
    st.write(f"**Moda:** {moda}")

def mostrar_grafica_variable_categorica(datos, variable_seleccionada):
    # Crear gráfica de barras para la variable categórica seleccionada
    conteo_categorias = datos[variable_seleccionada].value_counts()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=conteo_categorias.index, y=conteo_categorias.values)
    plt.xlabel(variable_seleccionada)
    plt.ylabel("Número de Ocurrencias")
    plt.xticks(rotation=45)
    st.pyplot()

def mostrar_grafica_serie_tiempo(datos, variable_seleccionada):
    fig = plt.figure(figsize=(10, 6))
    plt.plot(datos.index, datos[variable_seleccionada])
    plt.title(f"Serie de Tiempo para {variable_seleccionada}")
    plt.xlabel("Fecha")
    plt.ylabel(variable_seleccionada)
    st.pyplot(fig)

def mostrar_grafica_boxplot(datos, variable_categorica, variable_numerica_continua):
    if variable_categorica and variable_numerica_continua:
        
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=variable_categorica, y=variable_numerica_continua, data=datos)
        plt.xlabel(variable_categorica)
        plt.ylabel(variable_numerica_continua)
        plt.xticks(rotation=45)
        st.pyplot()

def mostrar_grafica_mosaico(datos, variable_categorica_1, variable_categorica_2):
    if variable_categorica_1 and variable_categorica_2:
        

        # Crear gráfico de mosaico con Plotly Express
        mosaic_data = pd.crosstab(datos[variable_categorica_1], datos[variable_categorica_2])
        fig = px.imshow(mosaic_data,
                        labels=dict(x=variable_categorica_2, y=variable_categorica_1),
                        x=mosaic_data.columns,
                        y=mosaic_data.index,
                        title=f"Mosaic Plot ({variable_categorica_1} vs {variable_categorica_2})")
        
        st.plotly_chart(fig)

        # Calcular el Coeficiente de Contingencia de Cramer
        contingency_table = pd.crosstab(datos[variable_categorica_1], datos[variable_categorica_2])
        chi2, _, _, _ = chi2_contingency(contingency_table)
        n = contingency_table.sum().sum()
        min_dim = min(contingency_table.shape) - 1
        cramer_v = np.sqrt(chi2 / (n * min_dim))

        st.write(f"Coeficiente de Contingencia de Cramer: {cramer_v:.4f}")



def mostrar_grafica_variables_seleccionadas(datos, variable_numerica_continua, variable_numerica_discreta, variable_fecha):
    # Mostrar gráfica de serie de tiempo para la variable de fecha seleccionada
    if variable_fecha:
        mostrar_grafica_serie_tiempo(datos, variable_fecha)

    # Mostrar otras gráficas según las variables seleccionadas
    if variable_numerica_continua and variable_numerica_discreta:
        st.subheader("Scatter Plot")
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=variable_numerica_continua, y=variable_numerica_discreta, data=datos)
        plt.xlabel(variable_numerica_continua)
        plt.ylabel(variable_numerica_discreta)
        st.pyplot()

def mostrar_archivo_cargado():
    st.title("Nueva Ventana - Archivo Cargado")
    # Mostrar el archivo cargado en la nueva ventana
    if hasattr(st.session_state, 'archivo_cargado') and st.session_state.archivo_cargado is not None:
        datos = None
        if st.session_state.archivo_cargado.name.endswith('.csv'):
            datos = pd.read_csv(st.session_state.archivo_cargado)
        elif st.session_state.archivo_cargado.name.endswith('.xlsx'):
            datos = pd.read_excel(st.session_state.archivo_cargado, engine='openpyxl')
        if datos is not None:
            st.write(datos)

            # Clasificar variables
            variables_categoricas, variables_numericas_continuas, variables_numericas_discretas, variables_fecha = clasificar_variables(datos)
            
            # Mostrar clasificación de variables
            st.subheader("Clasificación de Variables")
            st.write(f"**Variables Categóricas:** {', '.join(variables_categoricas)}")
            st.write(f"**Variables Numéricas Continuas:** {', '.join(variables_numericas_continuas)}")
            st.write(f"**Variables Numéricas Discretas:** {', '.join(variables_numericas_discretas)}")
            st.write(f"**Variables de Tipo Fecha:** {', '.join(variables_fecha)}")

            # Mostrar gráfica de variable numérica continua seleccionada
            st.subheader("Gráfica de densidad,  Variable Numérica Continua")

            variable_seleccionada_continua = st.selectbox("Selecciona una variable numérica continua:", variables_numericas_continuas)
            if variable_seleccionada_continua:
                mostrar_grafica_variable_numerica_continua(datos, variable_seleccionada_continua)

            # Mostrar gráfica de variable numérica discreta seleccionada
            st.subheader("Histograma,  Variable Numérica Discreta")
            variable_seleccionada_discreta = st.selectbox("Selecciona una variable numérica discreta:", variables_numericas_discretas)
            if variable_seleccionada_discreta:
                mostrar_grafica_variable_numerica_continua(datos, variable_seleccionada_discreta)

            # Mostrar gráfica de variable categórica seleccionada
            st.subheader("Gráfica de barras,  Variables Categóricas")
            variable_seleccionada_categorica = st.selectbox("Selecciona una variable categórica:", variables_categoricas)
            if variable_seleccionada_categorica:
                mostrar_grafica_variable_categorica(datos, variable_seleccionada_categorica)
            # Mostrar gráfica de scatter plot y métrica de correlación para variables numéricas seleccionadas
    st.subheader("Scatter Plot,  Variables Continuas")
    variable_seleccionada_numerica_1 = st.selectbox("Selecciona la primera variable numérica:", variables_numericas_continuas)
    variable_seleccionada_numerica_2 = st.selectbox("Selecciona la segunda variable numérica:", variables_numericas_continuas)

    if variable_seleccionada_numerica_1 and variable_seleccionada_numerica_2:
    # Scatter plot
        
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=variable_seleccionada_numerica_1, y=variable_seleccionada_numerica_2, data=datos)
        plt.xlabel(variable_seleccionada_numerica_1)
        plt.ylabel(variable_seleccionada_numerica_2)
        st.pyplot()

    # Métrica de correlación
        correlacion = datos[[variable_seleccionada_numerica_1, variable_seleccionada_numerica_2]].corr().iloc[0, 1]
        st.write(f"*Métrica de Correlación:* {correlacion:.2f}")
    else:
        st.warning("Por favor, carga un archivo para ver su contenido.")

    ##
    st.subheader("Series Temporales,  Variables Continuas/Discretas vrs Temporal")
    variable_seleccionada_tiempo_numerica = st.selectbox("Selecciona una variable numérica para la gráfica de tiempo:", variables_numericas_continuas + variables_numericas_discretas)
    variable_seleccionada_tiempo_fecha = st.selectbox("Selecciona una variable de fecha para la gráfica de tiempo:", variables_fecha)
    
    mostrar_grafica_tiempo(datos, variable_seleccionada_tiempo_numerica, variable_seleccionada_tiempo_fecha)

    ##
    st.subheader("BoxPlot,  Variables Continuas (Y) vrs Variables Categoricas (X)")
    variable_categorica_boxplot = st.selectbox("Selecciona una variable categórica para el Boxplot:", variables_categoricas)
    variable_numerica_continua_boxplot = st.selectbox("Selecciona una variable numérica continua para el Boxplot:", variables_numericas_continuas)

    mostrar_grafica_boxplot(datos, variable_categorica_boxplot, variable_numerica_continua_boxplot) 

    ##
    st.subheader("Grafica de Mosaico y Coeficiente de contingencia de Cramer, Variable categórica vrs Variable categórica")
    variable_categorica_1_mosaico = st.selectbox("Selecciona la primera variable categórica para el Mosaic Plot:", variables_categoricas)
    variable_categorica_2_mosaico = st.selectbox("Selecciona la segunda variable categórica para el Mosaic Plot:", variables_categoricas)

    mostrar_grafica_mosaico(datos, variable_categorica_1_mosaico, variable_categorica_2_mosaico)

def mostrar_grafica_tiempo(datos, variable_numerica, variable_fecha):
    if variable_numerica and variable_fecha:
        plt.figure(figsize=(10, 6))
        sns.lineplot(x=datos[variable_fecha], y=datos[variable_numerica])
        plt.xlabel(variable_fecha)
        plt.ylabel(variable_numerica)
        plt.xticks(rotation=45)
        st.pyplot()


def main():
    cargar_archivo()
    # Mostrar automáticamente la tabla y la clasificación de variables si un archivo es cargado
    mostrar_archivo_cargado()

if __name__ == '__main__':
    main()

