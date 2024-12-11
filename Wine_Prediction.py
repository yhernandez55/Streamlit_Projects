# Import libraries
import streamlit as st
import pandas as pd
import joblib
import nltk
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

# Project 3
def show_wine_predictions():
    # Input and prediction section 
    st.subheader('Prediction (1: Positive, 0: Negative)')
    input_review = st.text_input('Enter your wine review please')
    wine_model_path = os.path.join("Streamlit_Projects", "pickled_data","Wine_model.pkl")
    tfidf_vectorizer_path = os.path("Streamlit_Projects", "pickled_data","tfidf_Vectorizer_Wine.pkl")
    # Load trained model and TF-IDF vectorizer
    wine_model = joblib.load(wine_model_path)
    tfidf_vectorizer = joblib.load(tfidf_vectorizer_path)

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
    df_path = os.path.join("Streamlit_Projects", "DataSets","winemag-data-130k-v2.csv")
    df = pd.read_csv(df_path)
    df = df.drop(['Unnamed: 0', 'country', 'designation', 'points', 'price', 
              'province', 'region_1', 'region_2', 'variety'], axis=1)
    st.subheader('The actual df that is uncleaned')
    st.dataframe(df)

    df_cleaned_path = os.path.join("Streamlit_Projects", "pickled_data", "Cleaned_Wine_df.plk")
    df_cleaned = joblib.load(df_cleaned_path)
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
    eval_metrics_path = os.path.join("Streamlit_Projects", "pickled_data", "Evaluation_Metrics_Wine.pkl")
    accuracy, class_report = joblib.load(eval_metrics_path)
    st.subheader("Model Evaluation Metrics")
    st.write(f"Accuracy: {accuracy:.2f}")  # accuracy
    st.text("Classification Report:")  # class report from model
    st.text(class_report)
