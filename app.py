# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 22:53:46 2024

@author: CSU5KOR
"""


import os
import numpy as np
import nltk
import  pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import pickle
import streamlit as st


# Function to load the pre-trained model
@st.cache(allow_output_mutation=True)
def load_pretrained_model():
    try:
       feature_file='Updated_models/tfidf_scorer.pkl'
       with open(feature_file,'rb') as f:
           feature_extractor=pickle.load(f)
       f.close()
       
       encoder_file='Updated_models/encoder.pkl'
       with open(encoder_file,'rb') as f:
           encoder=pickle.load(f)
       f.close()
       
       model_file='Updated_models/classifier.pkl'
       with open(model_file,'rb') as f:
           model=pickle.load(f)
       f.close()
       return feature_extractor,encoder,model 
    except FileNotFoundError:
        st.error("Pre-trained model not found. Please make sure the model file exists.")
        st.stop()

# Streamlit App
st.title("Text Classification App(تطبيق تصنيف النص)")
st.write("This app demonstrates text classification using a pre-trained scikit-learn-based machine learning model.")
# Information about the app
st.sidebar.title("App Information")
st.sidebar.info(
    """This Streamlit app showcases text classification using a pre-trained scikit-learn-based 
     machine learning model on Arabic texts. The data is sourced is from 
     Arabic news articles organized into 3 balanced categories from www.alkhaleej.ae 
     Labels are categorized in: Medical,Sports,Tech.
     Enter text in the provided area, and the model will predict the label."""
)
# Load the pre-trained model
tfidf,encode,trained_model= load_pretrained_model()

# User input for text classification
user_text = st.text_area("Enter text for classification:")

# Classify user input
if user_text:
    tokens_new=nltk.wordpunct_tokenize(user_text)
    tokens_corrected=[i for i in tokens_new if len(i)>1]
    tfidf_tokens=' '.join(tokens_corrected)   

    x_test=tfidf.transform([tfidf_tokens])

    predicted=trained_model.predict(x_test)

    predicted_class=encode.inverse_transform(predicted)[0]
    
    st.write(f"Predicted Label: {predicted_class}")




