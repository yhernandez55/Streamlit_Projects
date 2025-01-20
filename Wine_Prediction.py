import os
import nltk
import numpy as np
import joblib
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import gdown

# Set the custom NLTK data directory
nltk_data_dir = os.path.join(os.path.expanduser("~"), "nltk_data")

# Create the directory if it doesn't exist
os.makedirs(nltk_data_dir, exist_ok=True)

# Append the custom NLTK data directory to NLTK's search path
nltk.data.path.append(nltk_data_dir)

# Force download missing NLTK resources to avoid errors
nltk.download("punkt", download_dir=nltk_data_dir, force=True)
nltk.download("stopwords", download_dir=nltk_data_dir, force=True)
nltk.download("wordnet", download_dir=nltk_data_dir, force=True)
nltk.download("omw-1.4", download_dir=nltk_data_dir, force=True)

# Verify the contents of the NLTK corpora directory
print(f"Contents of corpora directory: {os.listdir(os.path.join(nltk_data_dir, 'corpora'))}")

# File download URLs
file_urls = {
    "Wine_model.pkl": "https://drive.google.com/uc?id=1V6TMaNw6-6DaS4YhkwCLV6ofXrbfXNtg",
    "tfidf_Vectorizer_Wine.pkl": "https://drive.google.com/uc?id=1WV_1gxw2Uh26I12uJofxmAZeweVo3TSG",
    "Evaluation_Metrics_Wine.pkl": "https://drive.google.com/uc?id=1qqVdqlZpt7-fal9tEotIt0WkOJ_sTWBR",
    "Cleaned_Wine_df.plk": "https://drive.google.com/uc?id=1gXWCF9w-zAjNqJxQLw2wwQS16iSZAPqF",
    "winemag-data-130k-v2.csv": "https://drive.google.com/uc?id=1qd2rIjiqfx9dZ1q_aufpkwfZsl7u3DQb"
}

# Function to download files if not present locally
@st.cache_resource
def download_file(url, destination):
    if not os.path.exists(destination):
        gdown.download(url, destination, quiet=False)
    return destination

# Download all files
for file_name, file_url in file_urls.items():
    download_file(file_url, file_name)

# Cached resources loader
@st.cache_resource
def load_model(file_name):
    return joblib.load(file_name)

@st.cache_resource
def load_dataset(file_name):
    return pd.read_csv(file_name)

@st.cache_data
def load_cleaned_data(file_name):
    return joblib.load(file_name)

# Text preprocessing function
@st.cache_data
def preprocess_text(text):
    # Ensure NLTK resources are downloaded (only do this once)
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", download_dir=nltk_data_dir)

    # Process the text (tokenization, stopwords removal, lemmatization)
    stop_words = set(stopwords.words("english")).union({",", "."})
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)  # Tokenize
    tokens = [word.lower() for word in tokens if word.lower() not in stop_words]  # Remove stopwords
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]  # Lemmatize
    return ' '.join(lemmatized_tokens)  # Join tokens back into a string

# Main function to show wine predictions
def show_wine_predictions():
    # Load cached resources
    wine_model = load_model("Wine_model.pkl")
    tfidf_vectorizer = load_model("tfidf_Vectorizer_Wine.pkl")
    accuracy, class_report = load_model("Evaluation_Metrics_Wine.pkl")
    df_cleaned = load_cleaned_data("Cleaned_Wine_df.plk")
    raw_data = load_dataset("winemag-data-130k-v2.csv")

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

                # Predict sentiment
                wine_pred = wine_model.predict(input_review_transformed)
                if wine_pred[0] == 1:
                    st.success("This is a Positive review!")
                else:
                    st.warning("This is a Negative review!")
            except Exception as e:
                st.error(f"Error during transformation or prediction: {e}")
        else:
            st.warning("Please enter a valid review.")

    # Raw and cleaned data overview
    st.subheader("Dataset Overview")
    try:
        st.subheader('The Actual DataFrame (Uncleaned)')
        st.dataframe(raw_data.head(10))

        st.subheader('Cleaned DataFrame')
        st.dataframe(df_cleaned.head(10))
    except Exception as e:
        st.error(f"Error loading or processing dataframes: {e}")

    # Create and display wordcloud
    st.subheader("Wordcloud of Reviews")
    try:
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
    st.subheader("Model Evaluation Metrics")
    try:
        st.write(f"Accuracy: {accuracy:.2f}")
        st.text("Classification Report:")
        st.text(class_report)
    except Exception as e:
        st.error(f"Error displaying evaluation metrics: {e}")

# Run the Streamlit app
if __name__ == "__main__":
    show_wine_predictions()
