# Import libraries
import streamlit as st
import pandas as pd
import joblib
import nltk
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

