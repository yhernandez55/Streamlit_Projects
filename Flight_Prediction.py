# Import libraries
import os
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px
from xgboost import XGBRegressor


def show_flight_prediction():
    st.header('Predicting the Price of Flight')

    # Loading model and dataframe using relative paths
    pipeline_path = os.path.join('Streamlit_Projects', 'pickled_data', 'XGB_Model_Flight.pkl')
    pipeline = joblib.load(pipeline_path)
    df_path = os.path.join('Streamlit_Projects', 'DataSets', 'Airline_Clean_Dataset.csv')
    df = pd.read_csv(df_path)
    # Prediction section (Moved to the top)
    st.subheader('Predict Flight Price')

    # Dropdowns for categorical variables
    airline = st.selectbox('Airline', df['airline'].unique())
    source_city = st.selectbox('Source City', df['source_city'].unique())
    departure_time = st.selectbox('Departure Time', df['departure_time'].unique())
    stops = st.selectbox('Number of Stops', df['stops'].unique())
    arrival_time = st.selectbox('Arrival Time', df['arrival_time'].unique())
    destination_city = st.selectbox('Destination City', df['destination_city'].unique())
    flight_class = st.selectbox('Class', df['class'].unique())
    
    # Sliders for numerical variables
    duration = st.slider('Flight Duration (hours)', min_value=int(df['duration'].min()), max_value=int(df['duration'].max()), step=1)
    days_left = st.slider('Days Left until Departure', min_value=int(df['days_left'].min()), max_value=int(df['days_left'].max()), step=1)
    
    # for the predict button
    if st.button('Predict Flight Price'):
        # Define user input data for prediction
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
        
        # Display user input data
        st.write("User input data:", user_input_data)
        
        # Predict using the pipeline
        pred_airline = pipeline.predict(user_input_data)
        
        # Display the prediction
        st.success(f'The predicted price of the flight is: ${pred_airline[0]:.2f}')
    
    # Dataframe and stats
    st.subheader('Dataframe')
    df = pd.read_csv(df_path)
    st.dataframe(df)
    
    st.subheader('Stats')
    stats = df.describe()
    st.dataframe(stats)
    
    # Visuals
    st.subheader('Visuals')
    columns = st.sidebar.multiselect('Select columns to plot', df.columns)
    for col in columns:
        fig = px.histogram(df, x=col)
        st.plotly_chart(fig)
        
    # Load evaluation metrics (at the bottom of the app)
    eval_metrics_path = os.path.join("Streamlit_Projects", "pickled_data", "Evaluation_Metrics_Flight.pkl")
    eval_metrics = joblib.load('./Streamlit_Projects/pickled_data')  # Load the saved evaluation metrics
    rmse_train, rmse_test, fittest = eval_metrics  

    # Display evaluation metrics at the bottom
    st.subheader('Model Evaluation')
    st.write(f"Root Mean Squared Error (RMSE) on Training Data: {rmse_train:.2f}")
    st.write(f"Root Mean Squared Error (RMSE) on Test Data: {rmse_test:.2f}")
    st.write(f"Model Fit Status: {fittest[1]} (Ratio: {fittest[0]})")
    