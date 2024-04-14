import streamlit as st
import pickle

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer as ps
import string

# Load the model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title('Spam Email Classifier ')
input_mail= st.text_area("Enter the Email to verify... ")



# Preprocess the input
def transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)

    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)

    text=y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(ps().stem(i))

    return " ".join(y)

transformed_mail=transform_text(input_mail)

if st.button("Predict"):
    # Vectorize the input
    vector_input= tfidf.transform([transformed_mail])


    # Make predictions
    result= model.predict(vector_input)[0]


    # Display the result
    if result==1:
        st.header("Spam Email ")
    else:
        st.header("Not a Spam Email ")