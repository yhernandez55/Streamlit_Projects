# import libraries:
import os
import streamlit as st
import xgboost as xgb
# Display the XGBoost version
st.write("XGBoost version:", xgb.__version__)
import pandas as pd
import joblib
import plotly.express as px
import gdown
import sklearn

# Correct format URLs
file_urls = {
    "xgb_pipeline_Flight.pkl": "https://drive.google.com/uc?id=1AqBoCoK5e62MR4nyqREuZnENjAviLhe9",
    "Eval_Metrics_Flight.pkl": "https://drive.google.com/uc?id=1PErZ9vi18D3ORiTk0SScwpG5dkPSn7K_",
    "Airline_Clean_Dataset.csv": "https://drive.google.com/uc?id=1K0VmhNjhLvyR2H_F2kAyfBemH2p5z0oz"
}

# Function to download and get file from Google Drive
@st.cache_resource
def download_and_get_file(file_url, file_name):
    if not os.path.exists(file_name):  # Check if the file exists locally
        gdown.download(file_url, file_name, quiet=False)
    return file_name

# Cached dataset loader
@st.cache_data
def load_flight_dataset(file_name):
    df = pd.read_csv(file_name)
    # Optimize data types for memory efficiency
    df['stops'] = df['stops'].astype('category')
    df['airline'] = df['airline'].astype('category')
    df['source_city'] = df['source_city'].astype('category')
    df['destination_city'] = df['destination_city'].astype('category')
    df['class'] = df['class'].astype('category')
    return df

# Cached model loader
@st.cache_resource
def load_flight_model(file_name):
    return joblib.load(file_name)

def show_flight_prediction():
    # Download and load files
    flight_model_file = download_and_get_file(file_urls["xgb_pipeline_Flight.pkl"], "xgb_pipeline_Flight.pkl")
    eval_metrics_file = download_and_get_file(file_urls["Eval_Metrics_Flight.pkl"], "Eval_Metrics_Flight.pkl")
    dataset_file = download_and_get_file(file_urls["Airline_Clean_Dataset.csv"], "Airline_Clean_Dataset.csv")

    # Load the trained pipeline, evaluation metrics, and dataset
    pipeline = load_flight_model(flight_model_file)
    df = load_flight_dataset(dataset_file)
    eval_metrics = load_flight_model(eval_metrics_file)  # eval_metrics is also a pickle file

    # Force the XGBoost model to use CPU
    xgb_model = pipeline.named_steps['xgb_model']
    xgb_model.set_params(gpu_id=-1, tree_method='hist')  # Ensure CPU mode

    # Header and dataset overview
    st.header('Predicting the Price of Flights')

    # Dropdowns for categorical variables
    st.subheader('Predict Flight Price')
    airline = st.selectbox('Airline', df['airline'].unique())
    source_city = st.selectbox('Source City', df['source_city'].unique())
    departure_time = st.selectbox('Departure Time', df['departure_time'].unique())
    stops = st.selectbox('Number of Stops', df['stops'].unique())
    arrival_time = st.selectbox('Arrival Time', df['arrival_time'].unique())
    destination_city = st.selectbox('Destination City', df['destination_city'].unique())
    flight_class = st.selectbox('Class', df['class'].unique())

    # Sliders for numerical variables
    duration = st.slider('Flight Duration (hours)', 
                        min_value=int(df['duration'].min()), 
                        max_value=int(df['duration'].max()), 
                        step=1)
    days_left = st.slider('Days Left until Departure', 
                        min_value=int(df['days_left'].min()), 
                        max_value=int(df['days_left'].max()), 
                        step=1)
    
    # Prediction button
    if st.button('Predict Flight Price'):
        # Create input DataFrame for prediction
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
        # Debug: Display user input data
        st.write("User input data:", user_input_data)

        # Prediction
        try:
            predicted_price = pipeline.predict(user_input_data)
            st.success(f'The predicted price of the flight is: ${predicted_price[0]:.2f}')
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")

    # Dataset overview
    st.subheader('Dataset Overview')
    st.dataframe(df)

    # Display dataset statistics
    st.subheader('Dataset Statistics')
    st.dataframe(df.describe())

    # Visualization section
    st.subheader('Visualizations')
    columns = st.sidebar.multiselect('Select columns to visualize', df.columns)
    for col in columns:
        fig = px.histogram(df, x=col, title=f'Distribution of {col}')
        st.plotly_chart(fig)

    # Load and display evaluation metrics
    st.subheader('Model Evaluation Metrics')
    try:
        rmse_train, rmse_test, fittest = eval_metrics
        st.write(f"Root Mean Squared Error (Training): {rmse_train:.2f}")
        st.write(f"Root Mean Squared Error (Test): {rmse_test:.2f}")
        st.write(f"Model Fit Status: {fittest[1]} (Ratio: {fittest[0]})")
    except Exception as e:
        st.error(f"Error loading evaluation metrics: {str(e)}")