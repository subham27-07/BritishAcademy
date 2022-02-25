#APP
#checked
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

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
###############################Import LIWC############
# import liwc

# from collections import Counter
# from community import community_louvain
# from spacy.lang.en import English
# nlp = English()
# tokenizer = nlp.tokenizer
# import spacy 


# from pyvis import network as net
# from IPython.core.display import display, HTML

# spacy.cli.download("en_core_web_sm")
# spacy.load('en_core_web_sm')
# nlp = spacy.load("en_core_web_sm")

# import argparse
# parser = argparse.ArgumentParser()

# args = parser.parse_args()
############################################################

from pivottablejs import pivot_ui

st.title ("British Academy Data Analysis Tool")
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


#st.write(df.head()) #write new df Head(Top5)




# for pm in df["public_metrics"][0]:
#   df[pm] = df["public_metrics"].apply(lambda x:x[pm])

# df = df[(~df["clean_text"].isna())&(df["clean_text"].str.len()>0)&(df["clean_text"]!= "_url_")].copy()
# df_no_dup = df.drop_duplicates("clean_text")

# tokenizer = nlp.tokenizer
# # def getMoralCounts(text):
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

    fig = px.bar(x=[i[1] for i in hashtagCounts[:20]],y=[i[0] for i in hashtagCounts[:20]])

    st.write(fig)


#####################################################################





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

    fig = px.scatter(df, x=df['score'], y=df['created_at'], marginal_x="histogram", marginal_y="rug", width=700,height=900)

    st.plotly_chart(fig)


key=1
selectOptions=['Emotion Analysis','Sentiment Analysis' , 'Hastag Analysis', 'Topic Modelling', 'Emotion Analysis']

def addSelect():
    global key
    global selectOptions
    select= st.selectbox( '',selectOptions,key=str(key))
    key+=1

    selector(select)


def selector(select):
    global selectOptions

    if select == 'Topic Modelling':
        TopiModelling()
        ind=selectOptions.index('Topic Modelling')
        selectOptions.pop(ind)
        addSelect()
        
    elif select == 'Sentiment Analysis':
        Sentiment()
        ind=selectOptions.index('Sentiment Analysis')
        selectOptions.pop(ind)
        addSelect()
    
    elif select == 'Hastag Analysis':
        hastag()
        ind=selectOptions.index('Hastag Analysis')
        selectOptions.pop(ind)
        addSelect()
        

addSelect()
    
############################ Topic Modelling ################################


   





###################################################################################################



    

###################################################################################

###################################################################################################



###################################################################################




# t= pivot_ui(df)

# with open(t.src, encoding="utf8") as t:
#     components.html(t.read(), width=900, height=1000, scrolling=True)








###################################################################################









