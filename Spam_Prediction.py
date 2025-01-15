# Import necessary libraries
import os
import joblib
import gdown
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords

# Base directory setup
base_dir = os.path.dirname(os.path.abspath(__file__))

# Define folder paths
pickled_data_folder = os.path.join(base_dir, 'pickled_data')
datasets_folder = os.path.join(base_dir, 'DataSets')

# Ensure the required folders exist
os.makedirs(pickled_data_folder, exist_ok=True)
os.makedirs(datasets_folder, exist_ok=True)

# Define file paths
file_paths = {
    "Spam_Model.pkl": os.path.join(pickled_data_folder, "Spam_Model.pkl"),
    "tfidf_Vectorizer_Spam.pkl": os.path.join(pickled_data_folder, "tfidf_Vectorizer_Spam.pkl"),
    "Evaluation_Metrics_Spam.pkl": os.path.join(pickled_data_folder, "Evaluation_Metrics_Spam.pkl"),
    "spam.csv": os.path.join(datasets_folder, "spam.csv")
}

# Define file URLs for downloading
file_urls = {
    "Spam_Model.pkl": "https://drive.google.com/uc?id=17h6CL_3m22WtBrJQX7c5JMOhbbED1gNr",
    "tfidf_Vectorizer_Spam.pkl": "https://drive.google.com/uc?id=1Rp7KPNSO6q8KgUfSVQSxgVaryNmZn59k",
    "Evaluation_Metrics_Spam.pkl": "https://drive.google.com/uc?id=19AN9ZyqblDmwb3VzONajVkCVHi1fn-uZ"
}

# Project-1 (Spam Prediction)
def show_spam_project():
    st.header("Finding if a Message is a Spam")
    st.subheader("Spam is 1 and ham is 0")

    # Ensure files are available or download them
    for file_name, file_path in file_paths.items():
        if not os.path.exists(file_path):
            gdown.download(file_urls[file_name], file_path, quiet=False)
    
    # Assume all files are downloaded
    spam_model = joblib.load(file_paths["Spam_Model.pkl"])
    tfidf = joblib.load(file_paths["tfidf_Vectorizer_Spam.pkl"])
    assert hasattr(tfidf, 'idf_'), "TF-IDF Vectorizer is not fitted!"
    eval_metrics = joblib.load(file_paths["Evaluation_Metrics_Spam.pkl"])
 
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
    df_path = os.path.join("DataSets", "spam.csv")
    try:
        df = pd.read_csv(df_path, sep=',', encoding='latin-1')
        df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
        df = df[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'message'})
        df['label'] = df['label'].apply(lambda x: 1 if x == 'spam' else 0)
        st.dataframe(df)
    except FileNotFoundError:
        st.error("Spam dataset is missing. Please check the download link.") # checks to make sure there downloaded

    # Visuals: bar plot
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
