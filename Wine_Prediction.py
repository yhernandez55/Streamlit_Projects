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

# Define folder paths
pickled_data_folder = os.path.join(base_dir, 'pickled_data')
datasets_folder = os.path.join(base_dir, 'DataSets')

# Ensure the required folders exist
os.makedirs(pickled_data_folder, exist_ok=True)
os.makedirs(datasets_folder, exist_ok=True)

# File paths
file_paths = {
    "Wine_model.pkl": os.path.join(pickled_data_folder, 'Wine_model.pkl'),
    "tfidf_Vectorizer_Wine.pkl": os.path.join(pickled_data_folder, 'tfidf_Vectorizer_Wine.pkl'),
    "Evaluation_Metrics_Wine.pkl": os.path.join(pickled_data_folder, 'Evaluation_Metrics_Wine.pkl'),
    "Cleaned_Wine_df.plk": os.path.join(pickled_data_folder, 'Cleaned_Wine_df.plk'),
    "winemag-data-130k-v2.csv": os.path.join(datasets_folder, 'winemag-data-130k-v2.csv')
}

# File download URLs
file_urls = {
    "Wine_model.pkl": "https://drive.google.com/uc?id=1Ttx4EkgAAmqesIMBZw5wUQTlKik-l867",
    "tfidf_Vectorizer_Wine.pkl": "https://drive.google.com/uc?id=1zRRlf24PgYjWBtH7dPk39CDiAHCf4Ki3",
    "Evaluation_Metrics_Wine.pkl": "https://drive.google.com/uc?id=1BfgB_bO_9DUE-CyTqI3FdToKpqGKOAIK",
    "Cleaned_Wine_df.plk": " https://drive.google.com/uc?id=1fjIrotKhARgelSLh0SuGQQ_xxf63nya5",
    "winemag-data-130k-v2.csv": "https://drive.google.com/uc?id=1qd2rIjiqfx9dZ1q_aufpkwfZsl7u3DQb"
}


# Function to download files from Google Drive
def download_files():
    for file_name, url in file_urls.items():
        file_path = file_paths[file_name]
        if not os.path.exists(file_path):
            st.write(f"Downloading {file_name}...")
            try:
                gdown.download(url, file_path, quiet=False)
                st.success(f"Downloaded {file_name}")
            except Exception as e:
                st.error(f"Failed to download {file_name}: {e}")
        else:
            st.write(f"{file_name} already exists.")

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
        wine_model = joblib.load(file_paths["Wine_model.pkl"])
        tfidf_vectorizer = joblib.load(file_paths["tfidf_Vectorizer_Wine.pkl"])
        accuracy, class_report = joblib.load(file_paths["Evaluation_Metrics_Wine.pkl"])
        df_cleaned = joblib.load(file_paths["Cleaned_Wine_df.plk"])
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
        df = pd.read_csv(file_paths["winemag-data-130k-v2.csv"], encoding='ISO-8859-1')
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

    # Display predictions
    if __name__ == "__main__":
        show_wine_predictions()
