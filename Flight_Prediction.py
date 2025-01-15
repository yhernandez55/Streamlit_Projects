# import libraries:
import os
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import gdown

# Base directory setup
base_dir = os.path.dirname(os.path.abspath(__file__))

# Define folder paths
pickled_data_folder = os.path.join(base_dir, 'pickled_data')
datasets_folder = os.path.join(base_dir, 'DataSets')

# Ensure the required folders exist
os.makedirs(pickled_data_folder, exist_ok=True)
os.makedirs(datasets_folder, exist_ok=True)

# File paths
file_paths = {
    "xgb_pipeline_Flight.pkl": os.path.join(pickled_data_folder, 'xgb_pipeline_Flight.pkl'),
    "Eval_Metrics_Flight.pkl": os.path.join(pickled_data_folder, 'Eval_Metrics_Flight.pkl'),
    "Airline_Clean_Dataset.csv": os.path.join(datasets_folder, 'Airline_Clean_Dataset.csv')
}

# Correct format URLs
file_urls = {
    "xgb_pipeline_Flight.pkl": "https://drive.google.com/uc?id=10pWyBmDDgU0fUL6BBkbP4QZyvSo3zprn",
    "Eval_Metrics_Flight.pkl": "https://drive.google.com/uc?id=1nE6NlcdCDATh2PSf6aZ3IO4n3-sJANaC",
    "Airline_Clean_Dataset.csv": "https://drive.google.com/uc?id=1hZwlOyubjj5RembXMu4a5BMoYmQBw5o_"
}

# Use this to download the files with gdown
for file_name, file_url in file_urls.items():
    try:
        file_path = file_paths[file_name]  # Define file path for each file
        print(f"Downloading {file_name} from {file_url}...")
        gdown.download(file_url, file_path, quiet=False)  # Attempt to download the file
        print(f"{file_name} downloaded successfully.")  # Success message
    except Exception as e:
        print(f"Error downloading {file_name}: {e}")  # If error occurs, print error message

def show_flight_prediction():
    # Load the trained pipeline, eval metrics and dataset
    pipeline = joblib.load(file_paths["xgb_pipeline_Flight.pkl"])
    df = pd.read_csv(file_paths["Airline_Clean_Dataset.csv"])
    eval_metrics = joblib.load(file_paths["Eval_Metrics_Flight.pkl"])

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