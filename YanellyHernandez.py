import streamlit as st  # Import library

# Import each Python file
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
    Spam_Prediction.show_spam_project()  # Call the function from Spam_Prediction.py
elif choice == "Flight Prediction":
    Flight_Prediction.show_flight_prediction()  # Call the function from Flight_Prediction.py
elif choice == "Wine Predictions":
    Wine_Prediction.show_wine_predictions()  # Call the function from Wine_Prediction.py
