# import libraries:
import os
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import gdown
import sklearn

# Correct format URLs
file_urls = {
    "xgb_pipeline_Flight.pkl": "https://drive.google.com/uc?id=16y4-zCMv-Wz08U6N0jiOnYkaTLaGMQgf",
    "Eval_Metrics_Flight.pkl": "https://drive.google.com/uc?id=1yjc-det_BHdhMPrKotkENdfHyG32lke9",
    "Airline_Clean_Dataset.csv": "https://drive.google.com/uc?id=1K0VmhNjhLvyR2H_F2kAyfBemH2p5z0oz"
}

# Function to download files from Google Drive
def download_file(url, destination):
    try:
        # Try downloading the file using gdown
        print(f"Attempting to download from {url} to {destination}...")
        gdown.download(url, destination, quiet=False)
        print(f"Download successful: {destination}")
    except Exception as e:
        print(f"Error downloading file: {e}")
        raise

# Download files from Google Drive if not already present
for file_name, file_url in file_urls.items():
    if not os.path.exists(file_name):  # Check if the file already exists locally
        download_file(file_url, file_name)
    else:
        print(f"{file_name} already exists.")

# Streamlit app to show flight prediction
def show_flight_prediction():
    # Load the trained pipeline, eval metrics, and dataset
    pipeline = joblib.load("xgb_pipeline_Flight.pkl")
    xgb_model = pipeline.named_steps['xgb_model']

    # Load the dataset
    df = pd.read_csv("Airline_Clean_Dataset.csv")
    eval_metrics = joblib.load("Eval_Metrics_Flight.pkl")

    # Ensure no GPU-related parameters are being set (we set tree_method='hist' for CPU)
    xgb_model.set_params(tree_method='hist')  # Ensuring CPU mode

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