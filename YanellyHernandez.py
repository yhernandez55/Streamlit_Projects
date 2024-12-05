import streamlit as st # Import library

# Import each python file
import Homepage
import Spam_Prediction
import Flight_Prediction
import Wine_Prediction

# Sidebar navigation
st.sidebar.title("Navigation")
choice = st.sidebar.radio(
    "Go to",
    ["Homepage", "Spam Project", "Flight Prediction", "Wine Predictions"]
)

# Navigation logic
if choice == "Homepage":
    Homepage.show_homepage()  # Call the function from Homepage.py
elif choice == "Spam Project":
    Spam_Prediction.show_spam_project()  # Call the function from SpamProject.py
elif choice == "Flight Prediction":
    Flight_Prediction.show_flight_prediction()  # Call the function from FlightPrediction.py
elif choice == "Wine Predictions":
    Wine_Prediction.show_wine_predictions()  # Call the function from WinePredictions.py
