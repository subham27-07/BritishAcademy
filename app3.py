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

import re
#####################################################
from IPython.display import HTML, display



import moralizer
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

################### Redis Cache ################

###############################Import LIWC############

############################################################

from pivottablejs import pivot_ui

st.title ("Twitter Data Analysis Tool")
# st.sidebar.title("Analysis of Tweets")
st.markdown("This application is a Streamlit dashboard used to analyze sentiments, Emotions of tweets and Topic Modelling")
# st.sidebar.markdown("This application is a Streamlit dashboard used ""to analyze sentiments of tweets")

enc='utf-8'
spectra=st.file_uploader("upload file", type={"pkl",'txt'})
if spectra is not None:
    spectra_df=pd.read_pickle(spectra)
#st.write(spectra_df)


######################PICKLE###############################
#df = pd.read_pickle('clean_tweets_no_dup.pkl')

################################################

df=spectra_df[:500]
# st.write(df)
    


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
#########################################################################

# parse, category_names = liwc.load_token_parser('LIWC2007_English100131.dic')




# for pm in df["public_metrics"][0]:
#   df[pm] = df["public_metrics"].apply(lambda x:x[pm])

# df = df[(~df["clean_text"].isna())&(df["clean_text"].str.len()>0)&(df["clean_text"]!= "_url_")].copy()
# df_no_dup = df.drop_duplicates("clean_text")


# def getMoralCounts(text):
#   tokens = tokenizer(text)
#   tokens = [t.lower_ for t in tokens]
#   return dict(Counter(category for token in tokens for category in parse(token)))

# df_no_dup["moralFoundations"] = df_no_dup["clean_text"].apply(lambda x: getMoralCounts(x))

# for category in category_names:
#   df_no_dup[category] = df_no_dup["moralFoundations"].apply(lambda x: x[category] if category in x else 0)

# st.write(df_no_dup.head())


############################## Hastag Analysis ################


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


#####################################################################

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


######################################GeoCode#############################################
# geolocator = Nominatim(user_agent='twitter-analysis-cl')
# locs=df['user_location']
# geolocated = list(map(lambda x: [x,geolocator.geocode(x)[1] if geolocator.geocode(x) else None],locs))
# geolocated = pd.DataFrame(geolocated)
# geolocated.columns = ['locat','latlong']
# geolocated['lat'] = geolocated.latlong.apply(lambda x: x[0])
# geolocated['lon'] = geolocated.latlong.apply(lambda x: x[1])
# geolocated.drop('latlong',axis=1, inplace=True)


##################################################################################


key=1
selectOptions=['Network Analysis','Sentiment Analysis' , 'Hastag Analysis', 'Topic Modelling', 'Emotion Analysis', 'GeoCode']



def addSelect():
    global key
    global selectOptions
    select= st.selectbox( '',selectOptions,key=str(key))
    key+=1

    selector(select)


def selector(select):
    global selectOptions

    if select == 'Topic Modelling':
        st.markdown("Topic Models are very useful for the purpose for document clustering, organizing large blocks of textual data, information retrieval from unstructured text and feature selection.")
        st.image("Topic Model.png")
        st.markdown("Where the frequency of each word t is extracted for each class i and divided by the total number of words w. This action can be seen as a form of regularization of frequent words in the class. Next, the total, unjoined, number of documents m is divided by the total frequency of word t across all classes n.")
        TopiModelling()
        ind=selectOptions.index('Topic Modelling')
        selectOptions.pop(ind)
        addSelect()
        
    elif select == 'Sentiment Analysis':
        st.markdown("Sentiment analysis, also referred to as opinion mining, is an approach to natural language processing (NLP) that identifies the emotional tone behind a body of text. This is a popular way for organizations to determine and categorize opinions about a product, service, or idea.")
        st.image("full_nlp_pipeline.png")
        Sentiment()
        random_tweet = st.radio('Show Examples', ('POS', 'NEU', 'NEG'))
        st.markdown(df.query("label == @random_tweet")[["text"]].sample(n=1).iat[0, 0])
        ind=selectOptions.index('Sentiment Analysis')
        selectOptions.pop(ind)
        addSelect()
    
    elif select == 'Hastag Analysis':
        st.markdown("Hastag Analysis is used to measure the social media reach of hashtag campaign and its mentions. To measure social media engagement around your hashtag. To discover social media sentiment around a hashtag.")
        hastag()
        ind=selectOptions.index('Hastag Analysis')
        selectOptions.pop(ind)
        addSelect()

    elif select == 'Emotion Analysis':
        st.markdown("Emotion analysis is the process of identifying and analyzing the underlying emotions expressed in textual data. Emotion analytics can extract the text data from multiple sources to analyze the subjective information and understand the emotions behind it.")
        st.image("Emotion.png")
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
    
############################ Topic Modelling ################################


   





###################################################################################################



    

###################################################################################

###################################################################################################



###################################################################################




t= pivot_ui(df)

with open(t.src, encoding="utf8") as t:
    components.html(t.read(), width=900, height=1000, scrolling=True)








###################################################################################









