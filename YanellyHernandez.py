# Import libraries
import streamlit as st
import plotly.express as px
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score, classification_report
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import category_encoders as ce
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import xgboost as xgb
from xgboost import XGBRegressor
xgb_model = XGBRegressor()
rf = RandomForestClassifier()
from sklearn.metrics import root_mean_squared_error
import joblib
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS


Project1_tab, Project2_tab, Project3_tab = st.tabs(
    ['Spam Project',
     'Flight Prediction',
     'Wine Predictions'
     ]
     )


# Project-1
with Project1_tab:
    st.header("Finding if a Message is a Spam")
    st.subheader("Spam is 1 and ham is 0")

    # Load model and vectorizer
    spam_model = joblib.load("/Users/yanellyhernandez/Library/Mobile Documents/com~apple~CloudDocs/Desktop/Learning Fuze/mod2/week 8/spam_model.pkl")
    tfidf = joblib.load("/Users/yanellyhernandez/Library/Mobile Documents/com~apple~CloudDocs/Desktop/Learning Fuze/mod2/week 8/tfidf_vectorizer.pkl")

    # Making prediction 
    input_message = st.text_input("Enter your message please")
    if st.button('Predict'):
        if input_message:  # Check if a message was entered
            input_message_transform = tfidf.transform([input_message])
            pred = spam_model.predict(input_message_transform)
            if pred[0] == 1:
                st.error("This message is spam")  # Display result as spam
            else:
                st.success("This message is Not spam") 
            st.write(f"The predicted message is: {pred[0]}")

    # Loading and displaying data
    df = pd.read_csv('/Users/yanellyhernandez/Library/Mobile Documents/com~apple~CloudDocs/Desktop/Learning Fuze/mod2/week 8/spam.csv', sep=',', encoding='latin-1')
    df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
    df = df[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'message'})
    df['label'] = df['label'].apply(lambda x: 1 if x == 'spam' else 0) 
    st.dataframe(df)

    # Visuals: bar plot:
    spam_counts = df['label'].value_counts()
    fig, ax = plt.subplots()
    # Plotting the bar chart
    bars = ax.bar(['Ham (0)', 'Spam (1)'], spam_counts, color=['blue', 'green'])
    # Adding bar labels
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval, int(yval), va='bottom')  # Add label above the bar
    # Adding labels to axes
    ax.set_xlabel("Message Type")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Spam vs Ham")
    st.pyplot(fig)

    # Model evaluation metrics:
    accuracy, class_report = joblib.load('/Users/yanellyhernandez/Library/Mobile Documents/com~apple~CloudDocs/Desktop/Learning Fuze/mod2/week 8/evaluation_metrics.pkl')
    st.subheader("Model Evaluation Metrics")
    st.write(f"Accuracy: {accuracy:.2f}")  # accuracy
    st.text("Classification Report:")  # class report from model
    st.text(class_report)


# Project-2
with Project2_tab:
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

with Project3_tab:
    # Input and prediction section 
    st.subheader('Prediction (1: Positive, 0: Negative)')
    input_review = st.text_input('Enter your wine review please')

    # Load trained model and TF-IDF vectorizer
    wine_model = joblib.load('/Users/yanellyhernandez/Library/Mobile Documents/com~apple~CloudDocs/Desktop/Learning Fuze/mod2/week 11/Wine_model.pkl')
    tfidf_vectorizer = joblib.load('/Users/yanellyhernandez/Library/Mobile Documents/com~apple~CloudDocs/Desktop/Learning Fuze/mod2/week 11/tfidf_vectorizer.pkl')

    if st.button('Wine Predict'):
        # Define text preprocessing function
        def preprocess_text(text):
            stop_words = set(stopwords.words('english')).union({',', '.'})
            lemmatizer = WordNetLemmatizer()
            tokens = word_tokenize(text)  # Tokenize
            tokens = [word.lower() for word in tokens if word.lower() not in stop_words]  # Remove stopwords
            lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]  # Lemmatize
            return ' '.join(lemmatized_tokens)  # Join tokens back into a string
        
        # Preprocess the input review
        preprocessed_review = preprocess_text(input_review)
        # Transform the preprocessed review with the TF-IDF vectorizer
        input_review_transformed = tfidf_vectorizer.transform([preprocessed_review])
        # Make prediction
        wine_pred = wine_model.predict(input_review_transformed)
        # Display the prediction result
        if wine_pred[0] == 1:
            st.success('This is a Positive review!')
        else:
            st.warning('This is a Negative review.')

    # Initialize stopwords and lemmatizer 
    stop_words = set(stopwords.words('english')).union({',', '.'})
    lemmatizer = WordNetLemmatizer()

    # Header and data display
    st.header('Wine Predictions')
    st.subheader('Predicting Positive Reviews')

    # Load raw and cleaned data
    df = pd.read_csv("/Users/yanellyhernandez/Library/Mobile Documents/com~apple~CloudDocs/Desktop/Learning Fuze/mod2/week 7/winemag-data-130k-v2.csv")
    df = df.drop(['Unnamed: 0', 'country', 'designation', 'points', 'price', 
              'province', 'region_1', 'region_2', 'variety'], axis=1)
    st.subheader('The actual df that is uncleaned')
    st.dataframe(df)

    df_cleaned = joblib.load('/Users/yanellyhernandez/Library/Mobile Documents/com~apple~CloudDocs/Desktop/Learning Fuze/mod2/week 11/cleaned_df.plk')
    df_cleaned = df_cleaned.drop(['Unnamed: 0', 'country', 'designation', 'points', 
                              'price', 'province', 'region_1', 'region_2', 'variety'], axis=1)
    st.subheader('Cleaned Dataframe')
    st.dataframe(df_cleaned)

    # Create and display wordcloud
    st.subheader('Wordcloud of Reviews')
    df_cleaned['Cleaned_Lemma_Description'] = df_cleaned['Cleaned_Lemma_Description'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
    text_columns = ''.join(df_cleaned['Cleaned_Lemma_Description'])
    wordcloud = WordCloud(width=800, height=400, random_state=1).generate(text_columns)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

    # Model evaluation metrics:
    accuracy, class_report = joblib.load('/Users/yanellyhernandez/Library/Mobile Documents/com~apple~CloudDocs/Desktop/Learning Fuze/mod2/week 11/evaluation_metrics.pkl')
    st.subheader("Model Evaluation Metrics")
    st.write(f"Accuracy: {accuracy:.2f}")  # accuracy
    st.text("Classification Report:")  # class report from model
    st.text(class_report)

