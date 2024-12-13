import streamlit as st
import os

def show_homepage():
    # Display the image
    image_path = os.path.join("images", "Header.png")
    st.image(image_path, caption="Welcome to My Portfolio!", use_column_width=True)

    # Title and description
    st.title("Welcome to Yanelly's Portfolio")
    st.write("Explore my projects using the navigation sidebar!")
    
    st.write("""
    ### About Me
    I believe math tells stories, and data is its voice. As a passionate Data Scientist and Data Analyst, I love uncovering insights and solving problems through machine learning techniques and visual storytelling. 
    My goal is to turn data into actionable solutions that address a company's pain points and achieve meaningful impact. I’m equally enthusiastic about data visualization and statistical modeling to bring clarity to complex challenges.
    Explore my projects, and let's connect to collaborate on exciting problems!
    """)
