# Import necessary libraries
import os
import joblib
import gdown
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Define file URLs for downloading from Google Drive
file_urls = {
    "Spam_Model.pkl": "https://drive.google.com/uc?id=1F2-eTIaAs8C2l5dD17EhBWTaFotZAvkK", 
    "tfidf_Vectorizer_Spam.pkl": "https://drive.google.com/uc?id=1x7yaxTw6ytnW59m61Nn48h_5WkgGnh2h",
    "Evaluation_Metrics_Spam.pkl": "https://drive.google.com/uc?id=1rtIgfQ3XHGL88djwDbmvB_TKPp6hVZ4I",
    "spam.csv": "https://drive.google.com/uc?id=1QuR2MJhxOtqdAZz6WJ_9LaK2-zWs3vLS"
}

# Function to download files from Google Drive
@st.cache_resource
def download_file(url, destination):
    if not os.path.exists(destination):
        gdown.download(url, destination, quiet=False)
    return destination

# Download and cache all files
for file_name, file_url in file_urls.items():
    download_file(file_url, file_name)
    
# Load resources with caching
@st.cache_resource
def load_model(file_name):
    return joblib.load(file_name)

@st.cache_data
def load_dataset(file_name):
    return pd.read_csv(file_name, encoding="latin-1")
@st.cache_data
def preprocess_dataset(df):
    df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
    df = df[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'message'})
    df['label'] = df['label'].apply(lambda x: 1 if x == 'spam' else 0)
    return df

@st.cache_resource
def load_vectorizer(file_name):
    tfidf = joblib.load(file_name)
    if not hasattr(tfidf, 'idf_'):
        df = preprocess_dataset(load_dataset("spam.csv"))
        tfidf = TfidfVectorizer(stop_words=stopwords.words('english'))
        tfidf.fit_transform(df['message'])
        joblib.dump(tfidf, file_name)  # Save the fitted vectorizer
    return tfidf

# Spam Prediction function
def show_spam_project():
    st.header("Finding if a Message is Spam")
    st.subheader("Spam is 1 and ham is 0")

    # Load the model, vectorizer, and evaluation metrics
    spam_model = load_model("Spam_Model.pkl")
    tfidf = load_vectorizer("tfidf_Vectorizer_Spam.pkl")
    eval_metrics = load_model("Evaluation_Metrics_Spam.pkl")

    # Making prediction
    input_message = st.text_input("Enter your message please")
    if st.button('Predict'):
        if input_message:
            input_message_transform = tfidf.transform([input_message])
            pred = spam_model.predict(input_message_transform)
            if pred[0] == 1:
                st.error("This message is spam")
            else:
                st.success("This message is Not spam")
            st.write(f"The predicted message is: {pred[0]}")
        else:
            st.warning("Please enter a valid message.")

    # Load and preprocess dataset
    try:
        df = preprocess_dataset(load_dataset("spam.csv"))
        st.subheader("Dataset Overview")
        st.dataframe(df.head(10))
    except FileNotFoundError:
        st.error("Spam dataset is missing. Please check the download link.")
        return

    # Visualizations: bar plot
    spam_counts = df['label'].value_counts()
    fig, ax = plt.subplots()
    bars = ax.bar(['Ham (0)', 'Spam (1)'], spam_counts, color=['blue', 'green'])
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval, int(yval), va='bottom')
    ax.set_xlabel("Message Type")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Spam vs Ham")
    st.pyplot(fig)

    # Model evaluation metrics
    st.subheader("Model Evaluation Metrics")
    try:
        accuracy, class_report = eval_metrics
        st.write(f"Accuracy: {accuracy:.2f}")
        st.text("Classification Report:")
        st.text(class_report)
    except Exception as e:
        st.error(f"Error displaying evaluation metrics: {e}")