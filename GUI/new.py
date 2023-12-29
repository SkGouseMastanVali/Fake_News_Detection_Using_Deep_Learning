from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import sys
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from bnlp.corpus import punctuations, digits
from nltk.stem import PorterStemmer
from bnltk.tokenize import Tokenizers
 

#English Pre-processing
def clean_text(text):
    # Remove HTML tags using BeautifulSoup
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text()

    # Convert text to lowercase
    text = text.lower()

    # Remove special characters using regular expression
    text = re.sub(r"[^a-zA-Z0-9\s.,;:?!()\"'-]+", " ", text)

    # Tokenize text without including punctuation marks
    words = nltk.word_tokenize(text)

    # Remove punctuation marks from tokenized words
    words = [word for word in words if word not in string.punctuation]

    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    text = " ".join([word for word in words if word not in stop_words])

    return text

#Bangla Preprocessing
def clean_bangla(text):
    def punctuation_removal(text):
        all_list = [char for char in text if char not in (punctuations)]
        clean_str = ''.join(all_list)
        return clean_str

    def stopwordRemoval(text, stop_words):    
        l = text.split()
        stm = [elem for elem in l if elem not in stop_words]
        out = ' '.join(stm)
        return str(out)

    def digit_removal(text):
        all_list = [char for char in text if char not in digits]
        clean_str = ''.join(all_list)
        return clean_str

    text = punctuation_removal(text)
    text = digit_removal(text)

    data1 = pd.read_excel("E:\SE PROJECT FINAL\Final UI\stopwords_bangla.xlsx")
    stop = data1['words'].tolist()
    text = stopwordRemoval(text, stop)

    return text


app = Flask(__name__, static_folder='static')

# Load trained model
model_eng = load_model("E:\SE PROJECT FINAL\Final UI\english_model.h5")
model_beng = load_model("E:\SE PROJECT FINAL\Final UI\bangla_model.h5")

sys.stdout = open('output.txt', 'w')

@app.route('/')
def home():
    return render_template('main.html')

@app.route('/language_selection', methods=['POST'])
def language_selection():
    selected_language = request.form.get('language')
    
    if selected_language == 'english':
        return redirect(url_for('index'))
    elif selected_language == 'bengali':
        return redirect(url_for('index_bengali'))
    
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/index_bengali')
def index_bengali():
    return render_template('index_bengali.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['text']

        # Preprocess the input message
        new_message = clean_text(message)
    
        # Define list of news articles to predict
        news_articles = [new_message]
        
        # Tokenize and pad the news articles
        max_words = 6000
        max_length = 512
        tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
        tokenizer.fit_on_texts(news_articles)
        sequences = tokenizer.texts_to_sequences(news_articles)
        padded_sequences = pad_sequences(sequences, maxlen=max_length, padding="post", truncating="post")

        # Use trained model to predict whether news article is fake or real
        prediction = model_eng.predict(padded_sequences)
        y_pred_one = np.where(prediction >= 0.5, 1, 0)

        # Calculate percentages for fake and real news
        fake_percentage = (1 - prediction[0][0]) * 100
        real_percentage = prediction[0][0] * 100

        # Return prediction for current news article
        result = ""
        if y_pred_one == 0:
            result = f"The news article is predicted {fake_percentage:.2f}% FAKE and {real_percentage:.2f}% REAL."
        else:
            result = f"The news article is predicted {real_percentage:.2f}% REAL and {fake_percentage:.2f}% FAKE."

    return render_template('index.html', prediction=result)


@app.route('/predict_bengali', methods=['POST'])
def predict_bengali():
    if request.method == 'POST':
        data = request.form['text']
        # Preprocess the input message
        new_message = data
    
        # Define list of news articles to predict
        news_articles = [new_message]
        
        # Tokenize and pad the news articles
        max_words = 6000
        max_length = 512
        tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
        tokenizer.fit_on_texts(news_articles)
        sequences = tokenizer.texts_to_sequences(news_articles)
        padded_sequences = pad_sequences(sequences, maxlen=max_length, padding="post", truncating="post")

        # Use trained model to predict whether news article is fake or real
        prediction = model_beng.predict(padded_sequences)
        y_pred_one = np.where(prediction >= 0.5, 1, 0)

        # Calculate percentages for fake and real news
        fake_percentage = (1 - prediction[0][0]) * 100
        real_percentage = prediction[0][0] * 100

        # Return prediction for current news article
        result = ""
        if y_pred_one == 0:
            result = f"The news article is predicted {fake_percentage:.2f}% FAKE and {real_percentage:.2f}% REAL."
        else:
            result = f"The news article is predicted {real_percentage:.2f}% REAL and {fake_percentage:.2f}% FAKE."

    return render_template('index_bengali.html', prediction=result)


if __name__ == "__main__":
    app.run(debug=True, port=8080)
