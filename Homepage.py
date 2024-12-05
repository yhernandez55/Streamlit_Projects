import streamlit as st # Import library

def show_homepage():
    st.title("Welcome to Yanelly's Portfolio")
    st.write("Explore my projects using the navigation sidebar!")
    
    # Add more homepage content
    st.image("https://via.placeholder.com/600x200", caption="Your Portfolio Banner Here")
    st.write("""
    ### About Me
    I am a data scientist passionate about solving problems with machine learning and data visualization. 
    Check out the projects to see my work!
    """)



