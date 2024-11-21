import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn import datasets

def modifyData(lachman,cajon):
    def dividir_ventanas(X, ventana):
        longitud_señal = 500
        n_ventanas = longitud_señal // ventana
        X_ventaneado = X.reshape( n_ventanas, ventana)
        return X_ventaneado
    
    x=np.array(lachman)
    x_2=np.array(cajon)
    
    x_v = dividir_ventanas(x, 50)
    x_v_2 = dividir_ventanas(x_2, 50)
    indices_top_4 = [8,5,7,0]
    indices_top_4_2 = [9,0,8,2]
    # Extraer las 4 ventanas más importantes de la señal actual
    ventanas_top_4 = x_v[indices_top_4, :]
    ventanas_top_4_2 = x_v_2[indices_top_4_2, :]

    
    # Obtener la sumatoria, desviación estándar, media y mediana para cada ventana
    sumatorias = np.sum(ventanas_top_4, axis=1)
    desviaciones_std = np.std(ventanas_top_4, axis=1)
    medias = np.mean(ventanas_top_4, axis=1)
    medianas = np.median(ventanas_top_4, axis=1)
    sumatorias_2 = np.sum(ventanas_top_4_2, axis=1)
    desviaciones_std_2 = np.std(ventanas_top_4_2, axis=1)
    medias_2 = np.mean(ventanas_top_4_2, axis=1)
    medianas_2 = np.median(ventanas_top_4_2, axis=1)
    
    # Guardar las características para la señal actual
    caracteristicas =np.hstack((
        #sumatorias,
        desviaciones_std,
        medias,
        medianas,
        #sumatorias_2,
        desviaciones_std_2,
        medias_2,
        medianas_2
        )) 
    
    #caracteristicas_top_4.append(caracteristicas)
        
    caracteristicas_top_4=np.array(caracteristicas)
    return(caracteristicas_top_4)



# Title of the app
st.title('Clasificador de lesions de LCA')
# Cargar modelos desde archivos .pkl
def load_model(file_path):
    with open(file_path, 'rb') as file:
        model = joblib.load(file)
    return model

# Rutas a los archivos de modelos
model_paths = {
    "Modelo 1: RandomForest": "rf_pipeline.pkl",
    "Modelo 2: SVM": "svm_pipeline.pkl",
    "Modelo 3: Voting Ensamble": "vot_pipeline.pkl",
}

models = {name: load_model(path) for name, path in model_paths.items()}

# Selección del modelo
st.subheader("Selecciona un modelo:")
selected_model_name = st.radio("Elige un modelo:", list(models.keys()))
selected_model = models[selected_model_name]


# Mostrar imágenes asociadas al modelo seleccionado
image_paths = {
    "Modelo 1: RandomForest": ("assets/rfcr.jpg", "assets/rfcm.jpg"),
    "Modelo 2: SVM": ("assets/svmcr.jpg", "assets/svmcm.jpg"),
    "Modelo 3: Voting Ensamble": ("assets/vocr.jpg", "assets/vocm.jpg"),
}
st.subheader(f"Resultados de clasificación del {selected_model_name}")
image_1, image_2 = image_paths[selected_model_name]
col1, col2 = st.columns(2)
with col1:
    st.image(image_1, caption="Reporte de clasificación", use_column_width=True)
with col2:
    st.image(image_2, caption="Matriz de confusión", use_column_width=True)

# Entrada de datos por el usuario
st.subheader("Introduce los datos:")
vector_1_str = st.text_input("Datos maniobra de Lachman (separado por comas):", "1,2,3,4,5")
vector_2_str = st.text_input("Datos maniobra Cajon Anterior (separado por comas):", "5,4,3,2,1")

# Convertir las cadenas de texto a vectores numéricos
try:
    vector_1 = np.fromstring(vector_1_str, sep=",")
    vector_2 = np.fromstring(vector_2_str, sep=",")
except ValueError:
    st.error("Por favor, introduce vectores válidos en el formato correcto.")

# Procesar vectores con el pipeline
clases=['Negativa', 'Positiva']
if st.button("Clasificar"):
    try:
        # Concatenar vectores y pasarlos por el pipeline
        #combined_vector = np.hstack((vector_1, vector_2)).reshape(1, -1)
        processed_input = modifyData(vector_1, vector_2).reshape(1,-1)
        
        # Clasificar con el modelo seleccionado
        prediction = selected_model.predict(processed_input)
        st.success(f"La maniobra analizada resultó {clases[int(prediction[0])]} a una lesión de LCA.")
    except Exception as e:
        st.error(f"Error: {e}")