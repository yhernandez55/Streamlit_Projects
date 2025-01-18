# import libraries
import numpy as np
import joblib
import streamlit as st
import nltk
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import gdown

# Download necessary NLTK resources
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# 
# File download URLs
file_urls = {
    "Wine_model.pkl": "https://drive.google.com/uc?id=1V6TMaNw6-6DaS4YhkwCLV6ofXrbfXNtg",
    "tfidf_Vectorizer_Wine.pkl": "https://drive.google.com/uc?id=1WV_1gxw2Uh26I12uJofxmAZeweVo3TSG",
    "Evaluation_Metrics_Wine.pkl": "https://drive.google.com/uc?id=1qqVdqlZpt7-fal9tEotIt0WkOJ_sTWBR",
    "Cleaned_Wine_df.plk": "https://drive.google.com/uc?id=1gXWCF9w-zAjNqJxQLw2wwQS16iSZAPqF",
    "winemag-data-130k-v2.csv": "https://drive.google.com/uc?id=1qd2rIjiqfx9dZ1q_aufpkwfZsl7u3DQb"
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

# Function to preprocess text input
def preprocess_text(text):
    stop_words = set(stopwords.words("english")).union({",", "."})
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)  # Tokenize
    tokens = [word.lower() for word in tokens if word.lower() not in stop_words]  # Remove stopwords
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]  # Lemmatize
    return ' '.join(lemmatized_tokens)  # Join tokens back into a string

# Main function to show wine predictions
def show_wine_predictions():
    """
    This function will display the wine sentiment prediction page and handle the wine review input,
    preprocess the review, and make predictions using a pre-trained model.
    """
    # Load the necessary resources
    try:
        wine_model = joblib.load("Wine_model.pkl")
        tfidf_vectorizer = joblib.load("tfidf_Vectorizer_Wine.pkl")
        accuracy, class_report = joblib.load("Evaluation_Metrics_Wine.pkl")
        df_cleaned = joblib.load("Cleaned_Wine_df.plk")
    except Exception as e:
        st.error(f"Failed to load files: {e}")
        return

    # Streamlit App Title
    st.title("Wine Review Sentiment Analysis")

    # Input and prediction section
    st.subheader("Prediction (1: Positive, 0: Negative)")
    input_review = st.text_input("Enter your wine review please")

    if st.button("Wine Predict"):
        if input_review.strip():  # Check if the input is not empty
            # Preprocess input review
            preprocessed_review = preprocess_text(input_review)
            
            # Transform with TF-IDF using the existing vectorizer
            try:
                input_review_transformed = tfidf_vectorizer.transform([preprocessed_review])
                streamlit_features = set(tfidf_vectorizer.get_feature_names_out())

                # Ensure the number of features matches the model expectations
                if input_review_transformed.shape[1] != wine_model.n_features_in_:
                    st.error(f"Feature mismatch: Expected {wine_model.n_features_in_} features but got {input_review_transformed.shape[1]}")
                    return
                
                # Predict sentiment
                wine_pred = wine_model.predict(input_review_transformed)
                
                # Display prediction result
                if wine_pred[0] == 1:
                    st.success("This is a Positive review!")
                else:
                    st.warning("This is a Negative review!")
            except Exception as e:
                st.error(f"Error during transformation or prediction: {e}")
        else:
            st.warning("Please enter a valid review.")
            
    # Load raw and cleaned data
    try:
        df = pd.read_csv("winemag-data-130k-v2.csv")
        df = df.drop(['Unnamed: 0', 'country', 'designation', 'points', 'price', 
                      'province', 'region_1', 'region_2', 'variety'], axis=1)
        st.subheader('The Actual DataFrame (Uncleaned)')
        st.dataframe(df)

        df_cleaned = df_cleaned.drop(['Unnamed: 0', 'country', 'designation', 'points', 
                                      'price', 'province', 'region_1', 'region_2', 'variety'], axis=1)
        st.subheader('Cleaned DataFrame')
        st.dataframe(df_cleaned)
    except Exception as e:
        st.error(f"Error loading or processing dataframes: {e}")
        
    # Create and display wordcloud
    try:
        st.subheader('Wordcloud of Reviews')
        df_cleaned['Cleaned_Lemma_Description'] = df_cleaned['description'].apply(
            lambda x: ' '.join(x) if isinstance(x, list) else x
        )
        text_columns = ''.join(df_cleaned['Cleaned_Lemma_Description'])
        wordcloud = WordCloud(width=800, height=400, random_state=1).generate(text_columns)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error generating wordcloud: {e}")
        
    # Model evaluation metrics
    try:
        st.subheader("Model Evaluation Metrics")
        st.write(f"Accuracy: {accuracy:.2f}")
        st.text("Classification Report:")
        st.text(class_report)
    except Exception as e:
        st.error(f"Error displaying evaluation metrics: {e}")