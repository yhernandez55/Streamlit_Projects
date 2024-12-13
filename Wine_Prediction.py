# import libraries
import os
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import joblib
import gdown
import matplotlib.pyplot as plt
import streamlit as st

# Download necessary NLTK resources
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# Base directory setup
base_dir = os.path.dirname(os.path.abspath(__file__))

# File paths
model_path = os.path.join(base_dir, 'Streamlit_Projects', 'pickled_data', 'Wine_model.pkl')
tfidf_path = os.path.join(base_dir, 'Streamlit_Projects', 'pickled_data', 'tfidf_Vectorizer_Wine.pkl')
eval_metrics_path = os.path.join(base_dir, 'Streamlit_Projects', 'pickled_data', 'Evaluation_Metrics_Wine.pkl')
cleaned_df_path = os.path.join(base_dir, 'Streamlit_Projects', 'pickled_data', 'Cleaned_Wine_df.plk')
data_path = os.path.join(base_dir, 'Streamlit_Projects', 'DataSets', 'winemag-data-130k-v2.csv')

# File download URLs
file_urls = {
    model_path: "https://drive.google.com/uc?id=1l2bzhI4XwTRvNJ2RLuGPrJJ-YH4b0Uvr",
    tfidf_path: "https://drive.google.com/uc?id=1rvouN_GRd2BX4BvRnviSP0QmBAHOHYLl",
    eval_metrics_path: "https://drive.google.com/uc?id=1aCG-X8SE4teMbvaWaSPr5oE9QzB560hd",
    data_path: "https://drive.google.com/uc?id=1QuR2MJhxOtqdAZz6WJ_9LaK2-zWs3vLS",
    cleaned_df_path: "https://drive.google.com/uc?id=1CC_eXKfbw9_WZGwFM23KFETxozppbLIn"
}

# Function to download files from Google Drive
def download_files():
    for file_path, url in file_urls.items():
        if not os.path.exists(file_path):
            st.write(f"Downloading {os.path.basename(file_path)}...")
            gdown.download(url, file_path, quiet=False)

# Run the file download function
download_files()

# Function to preprocess text input
def preprocess_text(text):
    stop_words = set(stopwords.words("english")).union({",", "."})
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)  # Tokenize
    tokens = [word.lower() for word in tokens if word.lower() not in stop_words]  # Remove stopwords
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]  # Lemmatize
    return ' '.join(lemmatized_tokens)  # Join tokens back into a string

# Main function to display wine prediction page
def show_wine_predictions():
    """
    This function will display the wine sentiment prediction page and handle the wine review input,
    preprocess the review, and make predictions using a pre-trained model.
    """
    # Load required resources
    try:
        wine_model = joblib.load(model_path)
        tfidf_vectorizer = joblib.load(tfidf_path)
        accuracy, class_report = joblib.load(eval_metrics_path)
        df_cleaned = joblib.load(cleaned_df_path)
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
            # Transform with TF-IDF
            input_review_transformed = tfidf_vectorizer.transform([preprocessed_review])
            # Predict sentiment
            wine_pred = wine_model.predict(input_review_transformed)
            # Display prediction result
            if wine_pred[0] == 1:
                st.success("This is a Positive review!")
            else:
                st.warning("This is a Negative review!")
        else:
            st.warning("Please enter a valid review.")

    # Header and data display
    st.header("Wine Predictions")
    st.subheader("Predicting Positive Reviews")

    # Load raw data
    try:
        df = pd.read_csv(data_path, encoding='ISO-8859-1')
        st.subheader("Raw Data (Uncleaned)")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"Failed to load raw data: {e}")

    # Cleaned data display
    st.subheader("Cleaned Dataframe")
    try:
        df_cleaned = df_cleaned.drop(['Unnamed: 0', 'country', 'designation', 'points', 
                                      'price', 'province', 'region_1', 'region_2', 'variety'], axis=1)
        st.dataframe(df_cleaned.head())
    except Exception as e:
        st.error(f"Error in displaying cleaned dataframe: {e}")

    # WordCloud of reviews
    st.subheader("WordCloud of Reviews")
    try:
        df_cleaned['Cleaned_Lemma_Description'] = df_cleaned['Cleaned_Lemma_Description'].apply(
            lambda x: ' '.join(x) if isinstance(x, list) else x)
        text_columns = ' '.join(df_cleaned['Cleaned_Lemma_Description'])
        wordcloud = WordCloud(width=800, height=400, random_state=1).generate(text_columns)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error generating WordCloud: {e}")

    # Model evaluation metrics
    st.subheader("Model Evaluation Metrics")
    try:
        st.write(f"Accuracy: {accuracy:.2f}")
        st.text("Classification Report:")
        st.text(class_report)
    except Exception as e:
        st.error(f"Error displaying evaluation metrics: {e}")

