# Import libraries
import streamlit as st
import pandas as pd
import xgboost as xgb
from xgboost import XGBRegressor
xgb_model = XGBRegressor()
import joblib
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Project-2
def show_flight_prediction():
    st.header('Predicting the Price of Flight')
    # loading model and dataframe:
    pipeline = joblib.load('/Users/yanellyhernandez/Library/Mobile Documents/com~apple~CloudDocs/Desktop/Learning Fuze/mod2/week 10/xgb_model_plane.pkl') # model load
    df = pd.read_csv('/Users/yanellyhernandez/Library/Mobile Documents/com~apple~CloudDocs/Desktop/Learning Fuze/mod2/week1/AirlineRegression/Airline_Clean_Dataset.csv')
    
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
    df = pd.read_csv('/Users/yanellyhernandez/Library/Mobile Documents/com~apple~CloudDocs/Desktop/Learning Fuze/mod2/week1/AirlineRegression/Airline_Clean_Dataset.csv')
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
    eval_metrics = joblib.load('/Users/yanellyhernandez/Library/Mobile Documents/com~apple~CloudDocs/Desktop/Learning Fuze/mod2/week1/AirlineRegression/eval_metrics.pkl')  # Load the saved evaluation metrics
    rmse_train, rmse_test, fittest = eval_metrics  

    # Display evaluation metrics at the bottom
    st.subheader('Model Evaluation')
    st.write(f"Root Mean Squared Error (RMSE) on Training Data: {rmse_train:.2f}")
    st.write(f"Root Mean Squared Error (RMSE) on Test Data: {rmse_test:.2f}")
    st.write(f"Model Fit Status: {fittest[1]} (Ratio: {fittest[0]})")
