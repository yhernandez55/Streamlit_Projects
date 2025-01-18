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
    "Spam_Model.pkl": "https://drive.google.com/uc?id=1ohkTbAU3teOVpIB-en3X4TCM9ER0DsXa", 
    "tfidf_Vectorizer_Spam.pkl": "https://drive.google.com/uc?id=1tvmpktSYp3FZQSaJaS5zWg1CnJ_RKwtR",
    "Evaluation_Metrics_Spam.pkl": "https://drive.google.com/uc?id=1gI2-o8IDTxQKK4uzSNoeZqvTcUuRPgb3",
    "spam.csv": "https://drive.google.com/uc?id=1QuR2MJhxOtqdAZz6WJ_9LaK2-zWs3vLS"
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

# Project: Spam Prediction
def show_spam_project():
    st.header("Finding if a Message is Spam")
    st.subheader("Spam is 1 and ham is 0")

    # Load the model, vectorizer, and evaluation metrics
    spam_model = joblib.load("Spam_Model.pkl")
    tfidf = joblib.load("tfidf_Vectorizer_Spam.pkl")

    # In case the vectorizer is not fitted, re-fit it
    if not hasattr(tfidf, 'idf_'):
        st.warning("TF-IDF Vectorizer is not fitted! Re-fitting...")
        df = pd.read_csv("spam.csv", encoding="latin-1")
        df = df.rename(columns={'v1': 'label', 'v2': 'message'})
        df['label'] = df['label'].apply(lambda x: 1 if x == 'spam' else 0)
        tfidf = TfidfVectorizer(stop_words=stopwords.words('english'))
        X = tfidf.fit_transform(df['message'])
        joblib.dump(tfidf, "tfidf_Vectorizer_Spam.pkl")  # Save the re-fitted vectorizer

    eval_metrics = joblib.load("Evaluation_Metrics_Spam.pkl")

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

    # Data loading and displaying
    try:
        df = pd.read_csv("spam.csv", sep=',', encoding='latin-1')
        df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
        df = df[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'message'})
        df['label'] = df['label'].apply(lambda x: 1 if x == 'spam' else 0)
        st.dataframe(df)
    except FileNotFoundError:
        st.error("Spam dataset is missing. Please check the download link.")

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