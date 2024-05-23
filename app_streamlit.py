import streamlit as st
import joblib
import pandas as pd


modelo_path = './ML/modelo_entrenado.pkl'
columnas_path = './ML/columnas_entrenamiento.pkl'
# Cargar el modelo y las columnas de entrenamiento
modelo = joblib.load(modelo_path)
columnas_entrenamiento = joblib.load(columnas_path)

modelo_path_city = './ML/modelo_entrenado_city.pkl'
columnas_path_city = './ML/columnas_entrenamiento_city.pkl'
# Cargar el modelo y las columnas de entrenamiento para ciudad
modelo_city = joblib.load(modelo_path_city)
columnas_entrenamiento_city = joblib.load(columnas_path_city)

# Cargar las ciudades desde el archivo CSV
ciudades_df = pd.read_csv('./ML/ciudades.csv')
ciudades = ciudades_df['Ciudades'].tolist()

# Crear la interfaz de Streamlit
st.title("Predicción de Categorías de Restaurantes por estados")

# Pedir al usuario que ingrese los datos necesarios para la predicción
state = st.selectbox("Seleccione el estado", ["California", "New_York", "Florida", "Pennsylvania", "Nevada"])
stars = st.number_input("Ingrese la cantidad de estrellas", min_value=1.0, max_value=5.0, value=4.0, step=0.1)
sentimiento = st.number_input("Ingrese el valor de sentimiento", min_value=-1.0, max_value=1.0, value=0.1, step=0.1)
sentimiento_escalado = st.number_input("Ingrese el valor de sentimiento escalado", min_value=0.0, max_value=5.0, value=3.0, step=0.1)

# Realizar la predicción cuando el usuario haga clic en el botón
if st.button("Predecir"):
    # Crear un DataFrame con las características del restaurante
    input_data = pd.DataFrame({
        'state': [state],
        'stars': [stars],
        'sentimiento': [sentimiento],
        'sentimiento_escalado': [sentimiento_escalado]
    })

    # Aplicar codificación one-hot para el estado
    input_data = pd.get_dummies(input_data)

    # Asegurarse de que todas las columnas de estado estén presentes en el conjunto de datos de entrada
    # Si falta alguna columna de estado, agregarla con valor 0
    missing_cols = set(columnas_entrenamiento) - set(input_data.columns)
    for col in missing_cols:
        input_data[col] = 0

    # Reordenar las columnas para que coincidan con el orden del conjunto de entrenamiento
    input_data = input_data[columnas_entrenamiento]

    # Realizar la predicción
    predicted_category = modelo.predict(input_data)
    st.write(f"La categoría recomendada para abrir un restaurante con {stars} estrellas en {state} es: {predicted_category[0]}")


# Crear la interfaz de Streamlit para City
st.title("Predicción de Categorías de Restaurantes por ciudades")

# Pedir al usuario que ingrese los datos necesarios para la predicción
city_seleccionada = st.selectbox("Seleccione la ciudad", ciudades, key="city_select")
city_ingresada = st.text_input("O ingrese el nombre de la ciudad", key="city_input")
# Usar la ciudad ingresada si se proporciona, de lo contrario, usar la seleccionada
city = city_ingresada if city_ingresada else city_seleccionada

stars_city = st.number_input("Ingrese la cantidad de estrellas", min_value=1.0, max_value=5.0, value=4.0, step=0.1, key="stars_city")
sentimiento_city = st.number_input("Ingrese el valor de sentimiento", min_value=-1.0, max_value=1.0, value=0.1, step=0.1, key="sentimiento_city")
sentimiento_escalado_city = st.number_input("Ingrese el valor de sentimiento escalado", min_value=0.0, max_value=5.0, value=3.0, step=0.1, key="sentimiento_escalado_city")

# Realizar la predicción cuando el usuario haga clic en el botón
if st.button("Predecir Categoría por Ciudad"):
    # Crear un DataFrame con las características del restaurante
    input_data_city = pd.DataFrame({
        'city': [city],
        'stars': [stars_city],
        'sentimiento': [sentimiento_city],
        'sentimiento_escalado': [sentimiento_escalado_city]
    })

    # Aplicar codificación one-hot para la ciudad
    input_data_city = pd.get_dummies(input_data_city)

    # Asegurarse de que todas las columnas de ciudad estén presentes en el conjunto de datos de entrada
    missing_cols_city = set(columnas_entrenamiento_city) - set(input_data_city.columns)
    for col in missing_cols_city:
        input_data_city[col] = 0

    # Reordenar las columnas para que coincidan con el orden del conjunto de entrenamiento
    input_data_city = input_data_city[columnas_entrenamiento_city]

    # Realizar la predicción
    predicted_category_city = modelo_city.predict(input_data_city)
    st.write(f"La categoría recomendada para abrir un restaurante con {stars_city} estrellas en {city} es: {predicted_category_city[0]}")