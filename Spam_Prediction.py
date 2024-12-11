# Import necessary libraries
import os
import joblib
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Project-1 (Spam Prediction)
def show_spam_project():
    st.header("Finding if a Message is a Spam")
    st.subheader("Spam is 1 and ham is 0")

    # Load model and vectorizer using relative paths
    spam_model_path = os.path.join("Streamlit_Projects", "pickled_data", "Spam_Model.pkl")
    tfidf_path = os.path.join("Streamlit_Projects", "pickled_data", "tfidf_Vectorizer_Spam.pkl")
    spam_model = joblib.load(spam_model_path)
    tfidf = joblib.load(tfidf_path)

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
    df_path = os.path.join("Streamlit_Projects", "DataSets", "spam.csv")
    df = pd.read_csv(df_path, sep=',', encoding='latin-1')
    df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
    df = df[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'message'})
    df['label'] = df['label'].apply(lambda x: 1 if x == 'spam' else 0)
    st.dataframe(df)

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
    eval_metrics_path = os.path.join("Streamlit_Projects", "pickled_data", "Evaluation_Metrics_Spam.pkl")
    accuracy, class_report = joblib.load(eval_metrics_path)
    st.subheader("Model Evaluation Metrics")
    st.write(f"Accuracy: {accuracy:.2f}")
    st.text("Classification Report:")
    st.text(class_report)
