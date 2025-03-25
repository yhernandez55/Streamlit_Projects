# Streamlit_Projects

## Introduction--
Welcome to the Streamlit Projects repository! This mini-project collection demonstrates the power of interactive web applications for machine learning predictions using the Streamlit framework. Each project highlights the end-to-end process, from data preparation and model training to deployment.

## Project Descriptions--
  ### Spam Prediction: 
    1. Goal: Classify whether a message is spam or not.
    2. Input: Text message provided by the user.
    3. Output: Predict whether the message is spam or not.
    4. Key Tools: Natural Language Processing (NLP), Scikit-learn, RandomForestClassifier.
  ### Flight Price Prediction:
    1. Goal: Estimate flight prices based on user-selected features such as airline, departure time, and number of stops.
    2. Input: Nine features like source city, destination city, class, and flight duration.
    3. Output: The predicted price of the flight.
    4. Key Tools: XGBoost, category encoders, pipeline.
  ### Wine Review Prediction:
    1. Goal: Predict whether a wine review is positive or negative based on user input.
    2. Input: A text-based wine review.
    3. Output: Positive/Negative sentiment prediction.
    4. Key Tools: NLP pipelines, Scikit-learn, SMOTE, RandomForestClassifier.
    
## Technologies Used--
- Python
- Streamlit
- XGBoost
- Scikit-learn
- Pandas
- Numpy
- Plotly

## Getting Started--
  ### Prerequisites
  To explore the projects locally, follow these steps:
  - Python 3.11+ (recommended)
  - It is recommended that a virtual environment be used to avoid dependency conflicts.
    
  ### Live Demo
  You can view the live portfolio here: [Streamlit Portfolio](https://appprojects-gx243jxfrxhdqcye4agcdn.streamlit.app/)

  #### Hosted on Streamlit (Free Version):
  This app is hosted on Streamlit’s free tier, which means that it may go to sleep after a period of inactivity. If you visit the link and the app is in sleep mode, it will take a few seconds to wake up.

  
  ### Installation
  Alternatively, you can follow these steps to install it.
  ```bash
  # Clone the repository
  git clone https://github.com/yhernandez55/Streamlit_Projects.git
  
  # Navigate to the project directory
  cd Streamlit_Projects
  
  # Create and activate a virtual environment (optional but recommended)
  # On macOS/Linux
  python3 -m venv venv
  source venv/bin/activate
  
  # On Windows
  python -m venv venv
  venv\Scripts\activate
  
  # Install dependencies
  pip install -r requirements.txt
  
  # Run the Streamlit app
  streamlit run YanellyHernandez.py 


