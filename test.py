#APP
#checked
from linecache import cache
import streamlit as st
import pandas as pd 
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import plotly.express as px
import base64
import re
#####################################################
from IPython.display import HTML, display

import altair as alt

# import moralizer
######################################################


from copy import deepcopy
from bertopic import BERTopic
from transformers import pipeline

import networkx as nx


from datetime import datetime

from nltk.featstruct import _default_fs_class
from numpy import e
import streamlit as st

import streamlit.components.v1 as components
from PIL import Image
import pandas as pd
import tweepy as tw
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import altair as alt
import time
import setuptools
import pickle
import itertools
from collections import Counter
# from pyvis.network import Network
###############################Import LIWC############
import networkx as nx
# import moralizer
import regex as re
import altair as alt
from transformers import GPT2Tokenizer

from typing import NamedTuple


from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

from geopy.geocoders import Nominatim

################### Sentiment Example Library import ################
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax
import csv
import urllib.request
###############################Import LIWC############

############################################################

from pivottablejs import pivot_ui






# st.title ("Twitter Data Analysis Tool")
st.markdown("<h1 style='text-align: center; color: black;'>Twitter Data Analysis Tool</h1>", unsafe_allow_html=True)

# st.markdown("This application is a Streamlit dashboard used to analyze sentiments, Emotions of tweets and Topic Modelling")

st.markdown("<h3 style='text-align: center; color: black;'>This application is a Streamlit dashboard used to analyze sentiments,Hastag, Emotions of tweets and Topic Modelling</h3>", unsafe_allow_html=True)


enc='utf-8'
spectra=st.file_uploader("upload file", type={"csv",'txt'})
if spectra is not None:
    spectra_df=pd.read_csv(spectra)
#st.write(spectra_df)





######################PICKLE###############################
#df = pd.read_pickle('clean_tweets_no_dup.pkl')

################################################

df=spectra_df[:100]
# st.write(df)

# compression_opts = dict(method='zip',
#                         archive_name='out.csv') 
# df.to_csv('out.zip', index=False,
#           compression=compression_opts)  
##################################################################

#st.download_button(label='Download Current Result',data=df)
##################################################################
########################## Cleaning Texts #########################
def cleanTxt(text):
  text = re.sub(r'@[A-Za-z0-9]+','',text) #removed@mentions
  text = re.sub(r'#','',text)
  text = re.sub(r'RT[\s]','',text) 
  text = re.sub(r'https?:\/\/\S+','',text) #remove the hyperlinks

  return text

########################## Cleaning Texts ############################

########################## Apply clean on DF #########################
#spectra_df['body']=spectra_df['body'].apply(cleanTxt)
########################## Apply clean on DF #########################

from spacy.lang.en import English
from collections import Counter
from community import community_louvain
import re
import liwc
nlp = English()
tokenizer = nlp.tokenizer

import pydeck as pdk
#########################################################################

df2=df.dropna()
# st.write(df2)
x=df2[7:10]

############################# Geocoding #################################
import pandas as pd
from geopy.geocoders import Nominatim

# geolocator = Nominatim(user_agent="myApp")
# x[['location_lat', 'location_long']] = x['user_location'].apply(
#     geolocator.geocode).apply(lambda x: pd.Series(
#         [x.latitude, x.longitude], index=['location_lat', 'location_long']))


# d = {'lat': x['location_lat'], 'lon': x['location_long']}
# df3 = pd.DataFrame(data=d)

# st.map(df3)
# ################################# Map ########################


############################## Fixing Width ################



def _max_width_(prcnt_width:int = 70):
    max_width_str = f"max-width: {prcnt_width}%;"
    st.markdown(f""" 
                <style> 
                .reportview-container .main .block-container{{{max_width_str}}}
                </style>    
                """, 
                unsafe_allow_html=True,
    )


############################## Hastag Analysis ################
st.cache(suppress_st_warning=True)
def hastag():
    G = nx.DiGraph()
    for i,row in df.iterrows():
        source = row["user_name"]
        G.add_node(source)
        mentions = re.findall("\@([A-Za-z0-9_]+)", row["orig_text"])
        for mention in mentions:
            G.add_node(mention)
            if source not in G.neighbors(mention):
                G.add_edge(mention,source, weight=1)
            else:
                G[mention][source]["weight"] += 1

    nx.write_gexf(G, "mention_retweet.gexf")

    co_G = nx.Graph()
    allHashtagPairs = []
    for i,row in df.iterrows():
        hashtags = re.findall("#[A-Za-z0-9_]+", row["orig_text"])
        if len(hashtags) > 0:
            hashtags = [hashtag.lower() for hashtag in hashtags]
            hashtagCombination = itertools.combinations(hashtags,r=2)
            allHashtagPairs += hashtagCombination
    counts = Counter(allHashtagPairs)
    for source,target in counts:
        count = counts[(source,target)]
        co_G.add_edge(source,target, weight=count)


    nx.write_gexf(co_G, "hashtags.gexf")

    allHashtags = []
    for i in allHashtagPairs:
        for j in i:
            allHashtags.append(j)

    hashtagCounts = sorted(Counter(allHashtags).items(),key=lambda x: x[1],reverse=True)

    fig = px.bar(x=[i[1] for i in hashtagCounts[:20]],y=[i[0] for i in hashtagCounts[:20]],color=[i[1] for i in hashtagCounts[:20]])

    st.write(fig)


############################### Emotion Analysis ######################################

st.cache(suppress_st_warning=True)
def emotionAnalysis():
    emotion = pipeline('sentiment-analysis', model='arpanghoshal/EmoRoBERTa')
    def get_emotion_label(text):
        return(emotion(text)[0]['label'])

    df['clean_text'][1:10].apply(get_emotion_label)
    df['emotion'] = df['clean_text'].apply(get_emotion_label)
    # emotion_count = df['emotion'].value_counts()
    # emotion_count = pd.DataFrame({'Emotion':emotion.index, 'Tweets':emotion.values})

    fig = px.scatter(df, x=df['emotion'], y=df['created_at'], marginal_x="histogram", marginal_y="rug",color=df['emotion'], width=700,height=900)
    st.plotly_chart(fig)

# st.write(fig)

########################## Test for user input with example ################################



def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)
MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)
# PT
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# task='emotion'
# MODEL = f"cardiffnlp/twitter-roberta-base-{task}"

# tokenizer = AutoTokenizer.from_pretrained(MODEL)

# # download label mapping
# mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
# with urllib.request.urlopen(mapping_link) as f:
#     html = f.read().decode('utf-8').split("\n")
#     csvreader = csv.reader(html, delimiter='\t')
# labels = [row[1] for row in csvreader if len(row) > 1]

# # PT





##############################################################################




st.cache(suppress_st_warning=True)
def TopiModelling():
    global df
    model = BERTopic(language="english")

    docs = list(df['clean_text'].values)

    topics, probs = model.fit_transform(docs)

    model.get_topic_freq()
    x=model.get_topic(0)
    y=model.get_topic(2)
    #r=model.visualize_heatmap()
    s=model.visualize_barchart(top_n_topics=5,height=200,width=250)
    t=model.visualize_hierarchy(top_n_topics=50,height=1500)
    
    
    st.write(s)
    
    st.write(t)

    topic_freq = model.get_topic_info()
    topic_num_words_map = {row["Topic"]:row["Name"] for i,row in topic_freq.iterrows()}
    if probs is not None:
        df["topics"] = topics
        df["topic_probs"] = probs
        df["topic_words"] =  df["topics"].apply(lambda x: topic_num_words_map[x])


st.cache(suppress_st_warning=True)
def Sentiment():
    global df
    model_name = "finiteautomata/bertweet-base-sentiment-analysis"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

    df = (
        df
        .assign(sentiment = lambda x: x['clean_text'].apply(lambda s: classifier(s,truncation=True)))
        .assign(
            label = lambda x: x['sentiment'].apply(lambda s: (s[0]['label'])),
            score = lambda x: x['sentiment'].apply(lambda s: (s[0]['score']))
        )
    )

    fig = px.scatter(df, x=df['score'], y=df['created_at'], marginal_x="histogram", marginal_y="rug",color=df['label'], width=700,height=900)

    st.plotly_chart(fig)


    interval = alt.selection_interval()

    base = alt.Chart(df[::20]).mark_point().encode(
        y='created_at',
        color=alt.condition(interval, 'label', alt.value('lightgray')),
        tooltip='clean_text'
    ).add_selection(
        interval
    )

    hist = alt.Chart(df).mark_bar().encode(
        x='created_at',
        y='label',
        color='label'
    ).properties(
        width=800,
        height=80
    ).transform_filter(
        interval
    )

    st.altair_chart(base.mark_circle(color='label'), use_container_width=True)
    

    # st.altair_chart((scatter & hist), use_container_width=True) mark_line(color='firebrick')



##################################################################################


key=1

selectOptions=['Network Analysis','Sentiment Analysis' , 'Hastag Analysis', 'Topic Modelling', 'Emotion Analysis', 'GeoCode']


emotion = pipeline('sentiment-analysis', 
                    model='arpanghoshal/EmoRoBERTa')




def addSelect():
    global key
    global selectOptions
    with st.beta_expander("Add Analysis Tasks"):
        select= st.selectbox( '',selectOptions,key=str(key))
    key+=1

    selector(select)


def selector(select):
    global selectOptions

    if select == 'Topic Modelling':
        st.markdown("<h2 style='text-align: center; color: black;'>Topic Models are very useful for the purpose for document clustering, organizing large blocks of textual data, information retrieval from unstructured text and feature selection.</h2>", unsafe_allow_html=True)
        # st.markdown("Topic Models are very useful for the purpose for document clustering, organizing large blocks of textual data, information retrieval from unstructured text and feature selection.")
        col1, col2, col3 = st.beta_columns([1,6,1])
        with col2:
            st.image("Topic Model.png")
        # st.image("Topic Model.png")
        st.markdown("<h2 style='text-align: center; color: black;'>Where the frequency of each word t is extracted for each class i and divided by the total number of words w. This action can be seen as a form of regularization of frequent words in the class. Next, the total, unjoined, number of documents m is divided by the total frequency of word t across all classes n.</h2>", unsafe_allow_html=True)
        # st.markdown("Where the frequency of each word t is extracted for each class i and divided by the total number of words w. This action can be seen as a form of regularization of frequent words in the class. Next, the total, unjoined, number of documents m is divided by the total frequency of word t across all classes n.")
        result=st.button('Analysis',key=6)
        if result:
            TopiModelling()
        ind=selectOptions.index('Topic Modelling')
        selectOptions.pop(ind)
        addSelect()
        
    elif select == 'Sentiment Analysis':
        st.markdown("<h2 style='text-align: center; color: black;'>Sentiment analysis, also referred to as opinion mining, is an approach to natural language processing (NLP) that identifies the emotional tone behind a body of text. This is a popular way for organizations to determine and categorize opinions about a product, service, or idea.</h2>", unsafe_allow_html=True)
        # st.markdown("Sentiment analysis, also referred to as opinion mining, is an approach to natural language processing (NLP) that identifies the emotional tone behind a body of text. This is a popular way for organizations to determine and categorize opinions about a product, service, or idea.")
        st.write("Sentiment Analysis uses the Hugging Face Transformer to learn more about Hugging Face ???? [link](https://huggingface.co/docs/transformers/main_classes/pipelines)")
        # col1, col2, col3 = st.beta_columns([1,6,1])
        # with col2:
        #     st.image("full_nlp_pipeline.png")

        # st.image("full_nlp_pipeline.png")
        st.markdown("<h2 style='text-align: center; color: black;'>Test Our Model with Example.</h2>", unsafe_allow_html=True)
        text = st.text_input("")
        if text:
            text = preprocess(text)
            encoded_input = tokenizer(text, return_tensors='pt')
            output = model(**encoded_input)
            scores = output[0][0].detach().numpy()
            scores = softmax(scores)
            ranking = np.argsort(scores)
            ranking = ranking[::-1]
            for i in range(scores.shape[0]):
                l = config.id2label[ranking[i]]
                s = scores[ranking[i]]
                st.write(f"{i+1}) {l} {np.round(float(s), 4)}")
        st.markdown("<h2 style='text-align: center; color: black;'>If you are Satisfied with the result please go ahead and Analyze</h2>", unsafe_allow_html=True)
        result=st.button('Analysis',key=7)
        if result:
            Sentiment()
            random_tweet = st.radio('Show Examples', ('POS', 'NEU', 'NEG'))
            st.markdown(df.query("label == @random_tweet")[["text"]].sample(n=1).iat[0, 0])
        ind=selectOptions.index('Sentiment Analysis')
        selectOptions.pop(ind)
        addSelect()
    
    elif select == 'Hastag Analysis':
        st.markdown("<h2 style='text-align: center; color: black;'>Hastag Analysis is used to measure the social media reach of hashtag campaign and its mentions. To measure social media engagement around your hashtag. To discover social media sentiment around a hashtag.</h2>", unsafe_allow_html=True)
        # st.markdown("Hastag Analysis is used to measure the social media reach of hashtag campaign and its mentions. To measure social media engagement around your hashtag. To discover social media sentiment around a hashtag.")
        result=st.button('Analysis',key=8)
        if result:
            hastag()
        ind=selectOptions.index('Hastag Analysis')
        selectOptions.pop(ind)
        addSelect()

    elif select == 'Emotion Analysis':
        st.markdown("<h2 style='text-align: center; color: black;'>Emotion analysis is the process of identifying and analyzing the underlying emotions expressed in textual data. Emotion analytics can extract the text data from multiple sources to analyze the subjective information and understand the emotions behind it.</h2>", unsafe_allow_html=True)
        st.write("Emotion Analysis uses the Hugging Face Transformer to learn more about Hugging Face ???? [link](https://huggingface.co/docs/transformers/main_classes/pipelines)")
        # st.markdown("Emotion analysis is the process of identifying and analyzing the underlying emotions expressed in textual data. Emotion analytics can extract the text data from multiple sources to analyze the subjective information and understand the emotions behind it.")
        st.markdown("<h2 style='text-align: center; color: black;'>Test Our Model with Example.</h2>", unsafe_allow_html=True)
        # st.subheader('Test our Model with your input example')
        text1 = st.text_input("",key=2)
        if text1:
            text1 = preprocess(text1)
            # st.write(text1)
            output = emotion(text1)
            # st.write(output)

            for i in output:
                # st.write(i)
                new_list = list(i.values())
                # st.write(new_list)
                st.write(str(new_list)[1:-1])

        st.markdown("<h2 style='text-align: center; color: black;'>If you are Satisfied with the result please go ahead and Analyze</h2>", unsafe_allow_html=True)
        result=st.button('Analysis',key=9)
        if result:
            emotionAnalysis()
            random_tweet = st.radio('Shows Examples', ('amusement', 'anger', 'annoyance', 'confusion', 'disapproval', 'excitement', 'love', 'suprise'))
            st.markdown(df.query("emotion == @random_tweet")[["text"]].sample(n=1).iat[0, 0])
        ind=selectOptions.index('Emotion Analysis')
        selectOptions.pop(ind)
        addSelect()

    # elif select == 'GeoCode':
    #     geocode()
    #     ind=selectOptions.index('GeoCode')
    #     selectOptions.pop(ind)
    #     addSelect()
        

addSelect()
_max_width_()
    
############################ Topic Modelling ################################


   





###################################################################################################



    

###################################################################################

###################################################################################################



###################################################################################




t= pivot_ui(df)

# video_file = open('animation.gif', 'rb')
# video_bytes = video_file.read()

# st.video(video_bytes)

with st.beta_expander("Expand Me to see the DataFrame"):
    with open(t.src, encoding="utf8") as t:
        components.html(t.read(), width=700, height=1000, scrolling=True)
    

with st.beta_expander("Expand me to understand How to work with pivot table"):
    
    st.markdown("![Alt Text](https://pivottable.js.org/images/animation.gif)")





###################################################################################









