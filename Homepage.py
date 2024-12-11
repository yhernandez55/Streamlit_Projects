import streamlit as st
import os

def show_homepage():
    st.title("Welcome to Yanelly's Portfolio")
    st.write("Explore my projects using the navigation sidebar!")

    # Construct the path to the image in the Streamlit_Projects/images folder
    #header_image_path = os.path.join("Streamlit_Projects", "images", "header_image.png")

    # Display the image
    #st.image(header_image_path)

    st.write("""
    ### About Me
    I believe math tells stories, and data is its voice. As a passionate Data Scientist and Data Analyst, I love uncovering insights and solving problems through machine learning techniques and visual storytelling. 
    My goal is to turn data into actionable solutions that address a company's pain points and achieve meaningful impact. I’m equally enthusiastic about data visualization and statistical modeling to bring clarity to complex challenges.
    Explore my projects, and let's connect to collaborate on exciting problems!
    """)
