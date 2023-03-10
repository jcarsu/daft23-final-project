#!/usr/bin/env python
# coding: utf-8
# In[1]:
#Importamos biblioteca
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
#Importamos librerias de 'Natural Language ToolKit', NLTK, metodo VADER
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')
#Importar streamlit
import streamlit as st
# In[2]:
#Extraemos nuestra base de datos
f1=pd.read_csv('F1_tweetscleaned.csv')
# #Eliminamos columnas que no vamos a requerir para nuestro modelo
# f1= f1.drop(columns=['user_name','user_description','user_created','user_friends','user_favourites','is_retweet'], axis=1)
# #Rellenamos nulos con la moda y media
# f1['source']=f1['source'].replace(np.nan, 'Twitter for iPhone')
# f1['hashtags']=f1['hashtags'].replace(np.nan, np.mean)
# #Eliminamos nulos y verificamos nueva database
# f1=f1.dropna()
# #Reemplazamos valores redundantes
# f1['user_location']=(
#     f1['user_location'].replace('London','London, England').replace('UK','United Kingdom')
#     .replace('England, United Kingdom', 'United Kingdom'))
# #Reemplazamos valores redundantes
# f1['user_verified']=(
#     f1['user_verified'].replace('False',False).replace('True',True).replace('Its race week again #F1',False)
#     .replace('Precisely! #F1 https://t.co/HzNAMRzrnF',False)
#     .replace('Mexico City Grand Prix 2021 - F1 Race\r#MexicoGP 🇲🇽 #F1 \n🔴Go Live➡️https://t.co/6dTT0tNIwo\r\n🔴Go Live➡️https://t.co/6dTT0tNIwo\r\n\nSignup and watch unlimited https://t.co/OZO0Xuakz7',False)
#     .replace('Well done @alo_oficial 💙 #F1 #QatarGP',False))
f1sia=f1['date'],f1['text']
f1sia=pd.DataFrame(f1sia)
f1sia=f1sia.set_axis(['date', 'text'], axis=0)
f1sia=f1sia.transpose()
f1sia=f1sia.reset_index(drop=True)
#Llamamos al 'SentimentIntensityAnalyzer'
sia = SentimentIntensityAnalyzer()
f1sia['sentiment_score'] = f1sia['text'].apply(lambda x: sia.polarity_scores(x)['compound'])
#Probamos con cualquier texto no incluido en la bd
sentiment = SentimentIntensityAnalyzer()
# text_1 = "The book was a perfect balance between wrtiting style and plot."
# text_2 = "The car is yellow"
# sent_1 = sentiment.polarity_scores(text_1)
# sent_2 = sentiment.polarity_scores(text_2)
# print("Sentiment of text 1:", sent_1)
# print("Sentiment of text 2:", sent_2)
# In[3]:
st.title('Sentiment Analysis of a Text')
text_1 = st.text_input('Text Analyzer','Insert text')
if st.button('Analyze tweet'):
    sent_1 = sentiment.polarity_scores(text_1)
    if sent_1['compound'] >= 0.5:
        result='Positive'
    elif sent_1['compound'] < -0.5:
        result='Negative'
    else:
        result='Neutral'        
    st.write("Sentiment of text 1:", result)