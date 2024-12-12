from setuptools import setup, find_packages

setup(
    name="Streamlit Projects",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "streamlit",
        "joblib",
        "pandas",
        "matplotlib",
        "scikit-learn",
        "plotly",
        "nltk",
        "wordcloud",
        "gdown"  
    ],
    entry_points={
        "console_scripts": [
            "your_project_name=your_module_name:main_function",
        ],
    },
)

