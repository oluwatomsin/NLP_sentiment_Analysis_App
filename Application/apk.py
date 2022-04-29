import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import re
from gensim.utils import simple_preprocess

with open('./model&preprocessor/tfidf_dic.pkl', 'rb') as f:
    tfidf_dic = pickle.load(f)

with open('./model&preprocessor/text_classifier.pkl', 'rb') as f:
    text_classifier = pickle.load(f)


# Creating a text cleaning function
def text_cleaner(texts):
    # removing url
    new_texts = [re.sub("http\S+", "", str(token)) for token in texts]
    # removing mentions
    new_texts = [re.sub("@\S+", "", token) for token in new_texts]
    # further cleaning using gensim
    new_texts = [simple_preprocess(token, deacc=True) for token in new_texts]
    new_text_list = [' '.join(token) for token in new_texts]
    return new_text_list


# Creating the text analysis function
def sentiment_analysis(token):
    # Cleaning the text up
    new_list = list()
    new_list.append(token)
    new_token = text_cleaner(new_list)
    # Initializing the tfidf transformer
    transformer = TfidfTransformer()
    vec = TfidfVectorizer(max_features=2000, vocabulary=tfidf_dic)
    new_vec = transformer.fit_transform(vec.fit_transform(new_token))
    result = text_classifier.predict(new_vec)
    if result == 1:
        sent = "Positive"
    else:
        sent = "Negative"
    print(sent)
    return sent


# First form section
form = st.form(key="Page", clear_on_submit=False)
form.title('Sentiment Analysis Application')
form.title("User Info")
name = form.text_input("Whats is your name")
if form.form_submit_button("Welcome"):
    form.write(f"Welcome aboard, {name}. I hope you are excited to be here")

# Accepting the user text
st.title("Collecting User text and Analysing")
form_3 = st.form('sentiment', clear_on_submit=True)
form_3.write("Well!. All you have to do is give me a body of text and i will tell you if the text has a positive of "
             "negative sentiment")
text = form_3.text_input("Insert text here. Ensure that your text has a high polarity")
if form_3.form_submit_button("View Result"):
    st.write(f"Sentiment: {sentiment_analysis(text)}")
