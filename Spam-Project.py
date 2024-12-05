# Import libraries:
import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
import joblib
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer


# Project-1
def show_spam_project():
    st.header("Finding if a Message is a Spam")
    st.subheader("Spam is 1 and ham is 0")

    # Load model and vectorizer
    spam_model = joblib.load("/Users/yanellyhernandez/Library/Mobile Documents/com~apple~CloudDocs/Desktop/Learning Fuze/mod2/week 8/spam_model.pkl")
    tfidf = joblib.load("/Users/yanellyhernandez/Library/Mobile Documents/com~apple~CloudDocs/Desktop/Learning Fuze/mod2/week 8/tfidf_vectorizer.pkl")

    # Making prediction 
    input_message = st.text_input("Enter your message please")
    if st.button('Predict'):
        if input_message:  # Check if a message was entered
            input_message_transform = tfidf.transform([input_message])
            pred = spam_model.predict(input_message_transform)
            if pred[0] == 1:
                st.error("This message is spam")  # Display result as spam
            else:
                st.success("This message is Not spam") 
            st.write(f"The predicted message is: {pred[0]}")

    # Loading and displaying data
    df = pd.read_csv('/Users/yanellyhernandez/Library/Mobile Documents/com~apple~CloudDocs/Desktop/Learning Fuze/mod2/week 8/spam.csv', sep=',', encoding='latin-1')
    df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
    df = df[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'message'})
    df['label'] = df['label'].apply(lambda x: 1 if x == 'spam' else 0) 
    st.dataframe(df)

    # Visuals: bar plot:
    spam_counts = df['label'].value_counts()
    fig, ax = plt.subplots()
    # Plotting the bar chart
    bars = ax.bar(['Ham (0)', 'Spam (1)'], spam_counts, color=['blue', 'green'])
    # Adding bar labels
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval, int(yval), va='bottom')  # Add label above the bar
    # Adding labels to axes
    ax.set_xlabel("Message Type")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Spam vs Ham")
    st.pyplot(fig)

    # Model evaluation metrics:
    accuracy, class_report = joblib.load('/Users/yanellyhernandez/Library/Mobile Documents/com~apple~CloudDocs/Desktop/Learning Fuze/mod2/week 8/evaluation_metrics.pkl')
    st.subheader("Model Evaluation Metrics")
    st.write(f"Accuracy: {accuracy:.2f}")  # accuracy
    st.text("Classification Report:")  # class report from model
    st.text(class_report)



