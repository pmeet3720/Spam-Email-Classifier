import streamlit as st
import pickle

import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import PorterStemmer as ps
import string

tfidf = pickle.load(open('vectorizer1.pkl', 'rb'))
model = pickle.load(open('model1.pkl', 'rb'))

st.title('Spam Email Classifier ')
input_mail = st.text_area("Enter the Email to verify... ")

def preprocess_text(text):
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

transformed_mail=preprocess_text(input_mail)

if st.button("Predict"):
    vector_input= tfidf.transform([transformed_mail])

    result= model.predict(vector_input)[0]

    if result==1:
        st.header("Spam Email ")
    else:
        st.header("Not a Spam Email ")