# Import libraries
import os
import joblib
import gdown
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import nltk

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Download models and data files if missing
def download_Wine_models():
    # Wine_model.pkl
    if not os.path.exists('Streamlit_Projects/pickled_data/Wine_model.pkl'):
        gdown.download("https://drive.google.com/uc?id=1l2bzhI4XwTRvNJ2RLuGPrJJ-YH4b0Uvr", 
                       "Streamlit_Projects/pickled_data/Wine_model.pkl", quiet=False)
    
    # tfidf_Vectorizer_Wine.pkl
    if not os.path.exists('Streamlit_Projects/pickled_data/tfidf_Vectorizer_Wine.pkl'):
        gdown.download("https://drive.google.com/uc?id=1rvouN_GRd2BX4BvRnviSP0QmBAHOHYLl", 
                       "Streamlit_Projects/pickled_data/tfidf_Vectorizer_Wine.pkl", quiet=False)
    
    # Evaluation_Metrics_Wine.pkl
    if not os.path.exists('Streamlit_Projects/pickled_data/Evaluation_Metrics_Wine.pkl'):
        gdown.download("https://drive.google.com/uc?id=1aCG-X8SE4teMbvaWaSPr5oE9QzB560hd", 
                       "Streamlit_Projects/pickled_data/Evaluation_Metrics_Wine.pkl", quiet=False)
    
    # winemag-data-130k-v2.csv
    if not os.path.exists('Streamlit_Projects/DataSets/winemag-data-130k-v2.csv'):
        gdown.download("https://drive.google.com/uc?id=1QuR2MJhxOtqdAZz6WJ_9LaK2-zWs3vLS", 
                       "Streamlit_Projects/DataSets/winemag-data-130k-v2.csv", quiet=False)

# Run the function to download necessary files
download_Wine_models()

# Wine Predictions Project
def show_wine_predictions():
    st.header("Wine Reviews Prediction")
    st.subheader("Prediction (1: Positive, 0: Negative)")

    # Load model and TF-IDF vectorizer
    wine_model_path = os.path.join("Streamlit_Projects", "pickled_data", "Wine_model.pkl")
    tfidf_vectorizer_path = os.path.join("Streamlit_Projects", "pickled_data", "tfidf_Vectorizer_Wine.pkl")
    try:
        wine_model = joblib.load(wine_model_path)
        tfidf_vectorizer = joblib.load(tfidf_vectorizer_path)
    except FileNotFoundError:
        st.error("Required files are missing. Please check the download links or paths.") # checks to make sure there downloaded
        return

    # Input and prediction
    input_review = st.text_input("Enter your wine review:")
    if st.button("Wine Predict"):
        def preprocess_text(text):
            stop_words = set(stopwords.words('english')).union({',', '.'})
            lemmatizer = WordNetLemmatizer()
            tokens = word_tokenize(text)
            tokens = [word.lower() for word in tokens if word.lower() not in stop_words]
            lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
            return ' '.join(lemmatized_tokens)

        preprocessed_review = preprocess_text(input_review)
        input_review_transformed = tfidf_vectorizer.transform([preprocessed_review])
        prediction = wine_model.predict(input_review_transformed)

        if prediction[0] == 1:
            st.success("This is a Positive review!")
        else:
            st.warning("This is a Negative review.")

    # Load and display raw dataset
    st.subheader("Wine Dataset (Raw)")
    df_path = os.path.join("Streamlit_Projects", "DataSets", "winemag-data-130k-v2.csv")
    try:
        df = pd.read_csv(df_path)
        df = df[['description', 'title']]  # Keep only relevant columns
        st.dataframe(df.head())
    except FileNotFoundError:
        st.error("Wine dataset is missing. Please check the download link.") # checks to make sure there downloaded

    # Create and display WordCloud
    st.subheader("WordCloud of Reviews")
    try:
        df_cleaned_path = os.path.join("Streamlit_Projects", "pickled_data", "Cleaned_Wine_df.plk")
        df_cleaned = joblib.load(df_cleaned_path)
        all_text = ' '.join(df_cleaned['Cleaned_Lemma_Description'])
        wordcloud = WordCloud(width=800, height=400, random_state=1).generate(all_text)

        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)
    except FileNotFoundError:
        st.warning("Cleaned data for WordCloud is unavailable.") # checks to make sure there downloaded

    # Display evaluation metrics
    st.subheader("Model Evaluation Metrics")
    eval_metrics_path = os.path.join("Streamlit_Projects", "pickled_data", "Evaluation_Metrics_Wine.pkl")
    try:
        accuracy, class_report = joblib.load(eval_metrics_path)
        st.write(f"Accuracy: {accuracy:.2f}")
        st.text("Classification Report:")
        st.text(class_report)
    except FileNotFoundError:
        st.error("Evaluation metrics file is missing. Please check the download link.") # checks to make sure there downloaded