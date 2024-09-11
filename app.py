import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn import datasets




# Load the pipeline
pipeline = joblib.load('pipeline.pkl')

# Title of the app
st.title('Predecir el tipo de flor Iris')
iris = datasets.load_iris()
iris_df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])
# Display the DataFrame using Streamlit
st.write("Estos son los datos:")
st.dataframe(iris_df)
#Inputs 
st.write("Introduce la información de la flor:")
col1, col2 = st.columns(2)
with col1:
    sepalL = st.number_input("sepal width (cm)", min_value=0.0, step=0.1, format="%.1f")
    sepalW = st.number_input("sepal length (cm)", min_value=0.0,step=0.1, format="%.1f")
with col2:
    petalL = st.number_input("petal length (cm)", min_value=0.0, step=0.1, format="%.1f")
    petalW = st.number_input("petal width (cm)", min_value=0.0,  step=0.1, format="%.1f")

    


# Map the input into a pandas DataFrame
new_data = pd.DataFrame({
        'sepal length (cm)':[sepalL],
        'sepal width (cm)': [sepalW],
        'petal length (cm)': [petalL],
        'petal width (cm)': [petalW],
        })
dicto=['setosa', 'versicolor', 'virginica']
# When the user clicks the predict button
if st.button('Predecir'):
    # Make the prediction
    prediction = pipeline.predict(new_data)
    
    # Display the prediction
    st.write(f'La clase de la flor es: iris {dicto[int(prediction[0])]}')
