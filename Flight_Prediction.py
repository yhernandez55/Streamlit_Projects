import pandas as pd
import joblib
import streamlit as st
import gdown
import os
import plotly.express as px

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

def download_models():
    # XGB Model
    if not os.path.exists('Streamlit_Projects/pickled_data/XGB_Model_Flight.pkl'):
        gdown.download("https://drive.google.com/uc?id=1E7T5wc41VGcXWZXnXb-ES9pTgwtSU9_W", 
                       "Streamlit_Projects/pickled_data/XGB_Model_Flight.pkl", quiet=False)
    
    # Evaluation Metrics
    if not os.path.exists('Streamlit_Projects/pickled_data/Evaluation_Metrics_Flight.pkl'):
        gdown.download("https://drive.google.com/uc?id=1NE_iA88FlySd5Lwk3dr4w9IIV-TfxkCO", 
                       "Streamlit_Projects/pickled_data/Evaluation_Metrics_Flight.pkl", quiet=False)
    
    # Dataset
    if not os.path.exists('Streamlit_Projects/DataSets/Airline_Clean_Dataset.csv'):
        gdown.download("https://drive.google.com/uc?id=1hZwlOyubjj5RembXMu4a5BMoYmQBw5o_", 
                       "Streamlit_Projects/DataSets/Airline_Clean_Dataset.csv", quiet=False)

    
# Run the function to download files
download_models()

# Flight Price Prediction Application
def show_flight_prediction():
    st.header('Flight Price Prediction Application')

    # Load model and dataset
    pipeline_path = os.path.join('Streamlit_Projects', 'pickled_data', 'XGB_Model_Flight.pkl')
    df_path = os.path.join('Streamlit_Projects', 'DataSets', 'Airline_Clean_Dataset.csv')
    try:
        pipeline = joblib.load(pipeline_path)  # Load the pipeline (model + preprocessor)
        df = pd.read_csv(df_path)
    except FileNotFoundError:
        st.error("Required files are missing. Please check the download links or paths.")
        return

    # Prediction input
    st.subheader('Predict Flight Price')
    airline = st.selectbox('Airline', df['airline'].unique())
    source_city = st.selectbox('Source City', df['source_city'].unique())
    departure_time = st.selectbox('Departure Time', df['departure_time'].unique())
    stops = st.selectbox('Number of Stops', df['stops'].unique())
    arrival_time = st.selectbox('Arrival Time', df['arrival_time'].unique())
    destination_city = st.selectbox('Destination City', df['destination_city'].unique())
    flight_class = st.selectbox('Class', df['class'].unique())
    duration = st.slider('Flight Duration (hours)', min_value=int(df['duration'].min()), max_value=int(df['duration'].max()), step=1)
    days_left = st.slider('Days Left until Departure', min_value=int(df['days_left'].min()), max_value=int(df['days_left'].max()), step=1)

    if st.button('Predict Flight Price'):
        # Create user input DataFrame
        user_input_data = pd.DataFrame({
            'airline': [airline],
            'source_city': [source_city],
            'departure_time': [departure_time],
            'stops': [stops],
            'arrival_time': [arrival_time],
            'destination_city': [destination_city],
            'class': [flight_class],
            'duration': [duration],
            'days_left': [days_left]
        })

        # Ensure that the user input data matches the training data column order and structure
        try:
            user_input_transformed = pipeline.named_steps['preprocessor'].transform(user_input_data)
            # Get column names after transformation to match model expectations
            transformed_columns = pipeline.named_steps['preprocessor'].get_feature_names_out()
            # Create a DataFrame with transformed data to pass into the model for prediction
            user_input_transformed_df = pd.DataFrame(user_input_transformed, columns=transformed_columns)

            # Display input data
            st.write("Input Data for Prediction:", user_input_data)

            # Make prediction using the pipeline
            prediction = pipeline.predict(user_input_transformed_df)
            st.success(f'The predicted price of the flight is: ${prediction[0]:.2f}')
        except Exception as e:
            st.error(f"Error during prediction: {e}")

    # Dataframe and Statistics
    st.subheader('Flight Data')
    st.dataframe(df)

    st.subheader('Statistics Summary')
    st.dataframe(df.describe())

    # Visualization
    st.subheader('Visualizations')
    columns = st.sidebar.multiselect('Select Columns to Plot', df.columns)
    for col in columns:
        fig = px.histogram(df, x=col)
        st.plotly_chart(fig)

    # Evaluation Metrics
    st.subheader('Model Evaluation Metrics')
    eval_metrics_path = os.path.join('Streamlit_Projects', 'pickled_data', 'Evaluation_Metrics_Flight.pkl')
    try:
        rmse_train, rmse_test, fittest = joblib.load(eval_metrics_path)
        st.write(f"Root Mean Squared Error (RMSE) on Training Data: {rmse_train:.2f}")
        st.write(f"Root Mean Squared Error (RMSE) on Test Data: {rmse_test:.2f}")
        st.write(f"Model Fit Status: {fittest[1]} (Ratio: {fittest[0]})")
    except FileNotFoundError:
        st.warning("Evaluation metrics file is unavailable.")