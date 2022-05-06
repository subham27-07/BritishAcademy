#APP
#checked
from io import BytesIO
from pyxlsb import open_workbook as open_xlsb

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
from transformers import AutoTokenizer, AutoConfig
from transformers import AutoTokenizer
import numpy as np
from scipy.special import softmax
import csv
import urllib.request
###############################Import LIWC############

############################################################

from pivottablejs import pivot_ui
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode, JsCode
# from st_aggrid import AgGrid




new_title = '<h1 style="font-family:sans-serif;text-align: center; color:Black; font-size: 42px;">Twitter Data Analysis Tool</h1>'
st.markdown(new_title, unsafe_allow_html=True)

# st.title ("Twitter Data Analysis Tool")
# st.markdown("<h1 style='text-align: center; color: black;'>Twitter Data Analysis Tool</h1>", unsafe_allow_html=True)

st.markdown('<h3 style="font-family:sans-serif;text-align: center; color:Black;">This application is a Streamlit dashboard used to analyze sentiments,Hastag, Emotions of tweets and Topic Modelling</h3>', unsafe_allow_html=True)

# st.markdown("""
#         <style>
# @font-face {
#   font-family: 'Sans-Serif';
#   font-style: normal;
#   font-weight: 12;
#   src: url(https://fonts.gstatic.com/s/Sans-Serif/v12/IurY6Y5j_oScZZow4VOxCZZM.woff2) format('woff2');
#   unicode-range: U+0000-00FF, U+0131, U+0152-0153, U+02BB-02BC, U+02C6, U+02DA, U+02DC, U+2000-206F, U+2074, U+20AC, U+2122, U+2191, U+2193, U+2212, U+2215, U+FEFF, U+FFFD;
# }

#     html, body, [class*="css"]  {
#     font-family: 'Sans-Serif';
#     font-size: 16px;
#     }
#     </style>

#     """,
#         unsafe_allow_html=True,)

enc='utf-8'
spectra=st.file_uploader("upload file", type={"csv",'txt'})
if spectra is not None:
    spectra_df=pd.read_csv(spectra)
#st.write(spectra_df)


######################PICKLE###############################
#df = pd.read_pickle('clean_tweets_no_dup.pkl')

################################################

df=spectra_df[:1000]
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
x=df2

############################# Geocoding #################################
import pandas as pd
from geopy.geocoders import Nominatim


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
st.cache(suppress_st_warning=True,persist=True)
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

# st.cache(suppress_st_warning=True)
# def emotionAnalysis():
#     emotion = pipeline('sentiment-analysis', model='arpanghoshal/EmoRoBERTa')
#     def get_emotion_label(text):
#         return(emotion(text)[0]['label'])

#     df['clean_text'].apply(get_emotion_label)
#     df['emotion'] = df['clean_text'].apply(get_emotion_label)
#     # emotion_count = df['emotion'].value_counts()
#     # emotion_count = pd.DataFrame({'Emotion':emotion.index, 'Tweets':emotion.values})

#     fig = px.scatter(df, x=df['emotion'], y=df['created_at'], marginal_x="histogram", marginal_y="rug",color=df['emotion'], width=700,height=900)
#     st.plotly_chart(fig)


st.cache(suppress_st_warning=True,persist=True)
def emotionAnalysis():
    global df
    task='emotion'
    MODEL = f"cardiffnlp/twitter-roberta-base-{task}"
    emotion = pipeline('sentiment-analysis', model=MODEL)
    def get_emotion_label(text):
        return(emotion(text)[0]['label'])

    df['clean_text'][:4].apply(get_emotion_label)
    df['emotion'] = df['clean_text'].apply(get_emotion_label)


    emotion_count = df['emotion'].value_counts()
    emotion_count = pd.DataFrame({'emotion':emotion_count.index,'Tweets':emotion_count.values})
    # st.write(emotion_count)

    # figX = px.bar(emotion_count,x='emotion',y='Tweets',color='Tweets',height=500)
    # st.plotly_chart(figX)


    # figA1 = px.scatter(df_newA, x=df_newA['emotion'], y=df_newA['created_at'], hover_data=['clean_text'])
    # st.plotly_chart(figA1)



    figY = px.pie(emotion_count,values='Tweets',names='emotion')
    st.plotly_chart(figY)

    # emotion_count = df['emotion'].value_counts()
    # emotion_count = pd.DataFrame({'Emotion':emotion.index, 'Tweets':emotion.values})

    # figZ = px.scatter( df,x='emotion',y='created_at',color='emotion', hover_data=['clean_text'],width=700,height=900)
    
    # st.plotly_chart(figZ)

    fig8 = px.scatter(df, x=df['emotion'], y=df['created_at'], marginal_x="histogram", marginal_y="rug",color=df['emotion'],hover_data=['clean_text'], width=700,height=900)
    
    st.plotly_chart(fig8)
# st.write(fig)



############################################## Hate Sppech ################################

st.cache(suppress_st_warning=True,persist=True)
def hateAnalysis():
    global df
    task='hate'
    MODEL = f"cardiffnlp/twitter-roberta-base-{task}"
    hateSpeech = pipeline('sentiment-analysis', model=MODEL)
    def get_hate_label(text):
        return(hateSpeech(text)[0]['label'])

    df['clean_text'][:2].apply(get_hate_label)
    df['hate_Speech'] = df['clean_text'].apply(get_hate_label)

    hate_count = df['hate_Speech'].value_counts()
    hate_count = pd.DataFrame({'hate_Speech':hate_count.index,'Tweets':hate_count.values})
    # st.write(hate_count)

    # figX = px.bar(hate_count,x='hate_Speech',y='Tweets',color='Tweets',height=500)
    # st.plotly_chart(figX)

    figY = px.pie(hate_count,values='Tweets',names='hate_Speech')
    st.plotly_chart(figY, use_container_width=False)


    # figZ = px.scatter( df,x='hate_Speech',y='created_at',color='hate_Speech', hover_data=['clean_text'],width=700,height=900)
    
    # st.plotly_chart(figZ)

    figT = px.scatter(df, x=df['hate_Speech'], y=df['created_at'], marginal_x="histogram", marginal_y="rug",color=df['hate_Speech'],hover_data=['clean_text'], width=700,height=900)
    
    st.plotly_chart(figT)



    # figZ = px.scatter( df,x='emotion',y='created_at',color='emotion', hover_data=['clean_text'],width=700,height=900)
    
    # st.plotly_chart(figZ)

    # fig8 = px.scatter(df, x=df['emotion'], y=df['created_at'], marginal_x="histogram", marginal_y="rug",color=df['emotion'], width=700,height=900)
    
    # st.plotly_chart(fig8)
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

###################################### Emotion Analysis #############################
task='emotion'
MODEL = f"cardiffnlp/twitter-roberta-base-{task}"

tokenizer = AutoTokenizer.from_pretrained(MODEL)

# download label mapping
mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
with urllib.request.urlopen(mapping_link) as f:
    html = f.read().decode('utf-8').split("\n")
    csvreader = csv.reader(html, delimiter='\t')
labels = [row[1] for row in csvreader if len(row) > 1]

# PT
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

##############################################################################

###################################### Hate Speech #############################
task='hate'
MODEL = f"cardiffnlp/twitter-roberta-base-{task}"

tokenizer = AutoTokenizer.from_pretrained(MODEL)

# download label mapping
labels=[]
mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
with urllib.request.urlopen(mapping_link) as f:
    html = f.read().decode('utf-8').split("\n")
    csvreader = csv.reader(html, delimiter='\t')
labels = [row[1] for row in csvreader if len(row) > 1]

# PT
model = AutoModelForSequenceClassification.from_pretrained(MODEL)


##############################################################################


st.cache(suppress_st_warning=True)
def TopiModelling():
    global df
    model = BERTopic(language="english")

    docs = list(df['clean_text'].values)

    topics, probs = model.fit_transform(docs)
    vectorizer_model = CountVectorizer(ngram_range=(1, 3), stop_words="english")
    model.update_topics(docs, topics, vectorizer_model=vectorizer_model)

    model.get_topic_freq()
    x=model.get_topic(0)
    y=model.get_topic(2)
    r=model.visualize_heatmap()
    s=model.visualize_barchart(top_n_topics=10,height=200,width=250)
    t=model.visualize_hierarchy(top_n_topics=100,height=1500)
    
    
    st.write(s)
    
    st.write(t)
    st.write(r)

    topic_freq = model.get_topic_info()
    topic_num_words_map = {row["Topic"]:row["Name"] for i,row in topic_freq.iterrows()}
    if probs is not None:
        df["topics"] = topics
        df["topic_probs"] = probs
        df["topic_words"] =  df["topics"].apply(lambda x: topic_num_words_map[x])


st.cache(suppress_st_warning=True,persist=True)
def Sentiment():
    global df
    model_name = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    # model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # tokenizer = AutoTokenizer.from_pretrained(MODEL)
    config = AutoConfig.from_pretrained(MODEL)
    classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

    df = (
        df
        .assign(sentiment = lambda x: x['clean_text'].apply(lambda s: classifier(s,truncation=True)))
        .assign(
            label = lambda x: x['sentiment'].apply(lambda s: (s[0]['label'])),
            score = lambda x: x['sentiment'].apply(lambda s: (s[0]['score']))
        )
    )


    sentiment_count = df['label'].value_counts()
    sentiment_count = pd.DataFrame({'Sentiments':sentiment_count.index,'Tweets':sentiment_count.values})
    # st.write(sentiment_count)

    # fig1 = px.bar(sentiment_count,x='Sentiments',y='Tweets',color='Tweets',height=500)
    # st.plotly_chart(fig1)

    fig2 = px.pie(sentiment_count,values='Tweets',names='Sentiments')
    st.plotly_chart(fig2)

    ######################### altair ########################

    # scatter  = alt.Chart(df).mark_point().encode(x='created_at', y='label')

    # area = alt.Chart(df).mark_area(color="maroon").encode(x='created_at', y='score')

    # obj = alt.hconcat(scatter, area) #Horizontal Concatenation

    # st.altair_chart(obj)

  

    #########################################################
    # figW = px.scatter(df, x=df['score'], y=df['created_at'], color=df['label'],
    #              size=df['score'], hover_data=['clean_text'])

    df_new = df.loc[(df['label'] =='Negative') & (df['score']>0.6)]
    # st.write(df_new)

    df_new1 = df.loc[(df['label'] =='Positive') & (df['score']>0.5)]
    # st.write(df_new1)

    df_new2 = df.loc[(df['label'] =='Neutral') & (df['score']>0.5)]
    # st.write(df_new2)




    figW = px.scatter(df_new, x=df_new['score'], y=df_new['created_at'], color=df_new['label'],
                 size=df_new['score'], hover_data=['clean_text'])
    st.plotly_chart(figW)

    figQ = px.scatter(df_new1, x=df_new1['score'], y=df_new1['created_at'],
                 size=df_new1['score'], hover_data=['clean_text'])
    st.plotly_chart(figQ)

    figR = px.scatter(df_new2, x=df_new2['score'], y=df_new2['created_at'], color=df_new2['label'],
                 size=df_new2['score'], hover_data=['clean_text'])
    st.plotly_chart(figR)

    # for i in range(1, 3):
    #     cols = st.beta_columns(3)
    #     cols[0].write(st.plotly_chart(figW))
    #     cols[1].write(st.plotly_chart(figQ))
    #     cols[2].write(st.plotly_chart(figR))
        

    #########################################################

    # fig = px.scatter(df, x=df['score'], y=df['created_at'], marginal_x="histogram", marginal_y="rug",color=df['label'], width=700,height=900)
    # fig3 = px.scatter( df,x='label',y='created_at',color='label',size='score', hover_data=['clean_text'],width=700,height=900)

    # st.plotly_chart(fig3)
    # fig = px.scatter(df, x=df[df['label']=='Negative'], y=df['score'], width=700,height=900)

   
    

    figP = px.scatter(df, x=df['label'], y=df['created_at'], marginal_x="histogram", marginal_y="rug",color=df['label'],size='score',hover_data=['clean_text'], width=700,height=900)
    
    st.plotly_chart(figP)
    



##################################################################################


key=1

selectOptions=['Add Analysis Tasks','Sentiment Analysis' ,'Hate Speech Analysis' , 'Hastag Analysis', 'Topic Modelling', 'Emotion Analysis']


emotion = pipeline('sentiment-analysis', 
                    model='arpanghoshal/EmoRoBERTa')

def to_excel(df):
    output=BytesIO()


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
        with st.beta_expander("Expand Me to see Topic Modelling Analysis"):
            st.markdown('<p style="font-family:sans-serif;text-align: left; color:Black;font-size: 16px;">Topic Models are very useful for the purpose for document clustering, organizing large blocks of textual data, information retrieval from unstructured text and feature selection.</p>', unsafe_allow_html=True)
            col1, col2, col3 = st.beta_columns([1,6,1])
            with col2:
                st.image("Topic Model.png")
            # st.image("Topic Model.png")
            st.markdown('<p style="font-family:sans-serif;text-align: left; color:Black;font-size: 16px;">Where the frequency of each word t is extracted for each class i and divided by the total number of words w. This action can be seen as a form of regularization of frequent words in the class. Next, the total, unjoined, number of documents m is divided by the total frequency of word t across all classes n.</p>', unsafe_allow_html=True)
            # result=st.button('Analyze',key=6)
            # if result:
            TopiModelling()
        ind=selectOptions.index('Topic Modelling')
        selectOptions.pop(ind)
        addSelect()
       
    elif select == 'Sentiment Analysis':
        with st.beta_expander("Expand Me to see the Sentiment Analysis"):
            st.markdown('<p style="font-family:sans-serif;text-align: left; color:Black;font-size: 16px;">Sentiment analysis, also referred to as opinion mining, is an approach to natural language processing (NLP) that identifies the emotional tone behind a body of text. This is a popular way for organizations to determine and categorize opinions about a product, service, or idea.</p>', unsafe_allow_html=True)
            st.write("Sentiment Analysis uses the Hugging Face Transformer to learn more about Hugging Face 🤗 [link](https://huggingface.co/docs/transformers/main_classes/pipelines)")
            # col1, col2, col3 = st.beta_columns([1,6,1])
            # with col2:
            #     st.image("full_nlp_pipeline.png")

            # st.image("full_nlp_pipeline.png")
            st.markdown('<p style="font-family:sans-serif;text-align: center; color:Black;font-size: 18px;">Test the Model with Example.</p>', unsafe_allow_html=True)
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
            # st.markdown('<p style="font-family:sans-serif;text-align: center; color:Black;font-size: 18px;">If you are Satisfied with the result please go ahead and Analyze</p>', unsafe_allow_html=True)
            
            # result=st.button('Analysis',key=7)
            # if result:
            Sentiment()
                # random_tweet = st.radio('Show Examples', ('POS', 'NEU', 'NEG'))
                # st.markdown(df.query("label == @random_tweet")[["text"]].sample(n=1).iat[0, 0])
        ind=selectOptions.index('Sentiment Analysis')
        selectOptions.pop(ind)
        addSelect()
    
    elif select == 'Hastag Analysis':
        with st.beta_expander("Expand Me to see the Hastag Analysis"):
            st.markdown('<p style="font-family:sans-serif;text-align: left; color:Black;font-size: 18px;">Hastag Analysis is used to measure the social media reach of hashtag campaign and its mentions. To measure social media engagement around your hashtag. To discover social media sentiment around a hashtag.</p>', unsafe_allow_html=True)
            result=st.button('Analyze',key=8)
            if result:
                hastag()
        ind=selectOptions.index('Hastag Analysis')
        selectOptions.pop(ind)
        addSelect()

    elif select == 'Hate Speech Analysis':
        with st.beta_expander("Expand Me to see the HateSpeech Analysis"):
            st.markdown('<p style="font-family:sans-serif;text-align: left; color:Black;font-size: 18px;">Hate Speech in the form of racist and sexist remarks are a common occurance on social media.“Hate speech is defined as any communication that disparages a person or a group on the basis of some characteristics such as race, color, ethnicity, gender, sexual orientation, nationality, religion, or other characteristic.</p>', unsafe_allow_html=True)
            st.write("This is a roBERTa-base model trained on ~58M tweets and finetuned for hate speech detection with the TweetEval benchmark. 🤗 [link](https://huggingface.co/cardiffnlp/twitter-roberta-base-hate?text=I+like+you.+I+love+you)")
            st.markdown('<p style="font-family:sans-serif;text-align: center; color:Black;font-size: 18px;">Test the Model with Example.</p>', unsafe_allow_html=True)
            
            textA = st.text_input("",key=3)
            if textA:
                text = preprocess(textA)
                encoded_input = tokenizer(text, return_tensors='pt')
                output = model(**encoded_input)
                scores = output[0][0].detach().numpy()
                scores = softmax(scores)
                ranking = np.argsort(scores)
                ranking = ranking[::-1]
                for i in range(scores.shape[0]):
                    l = labels[ranking[i]]
                    s = scores[ranking[i]]
                    st.write(f"{i+1}) {l} {np.round(float(s), 4)}")
            st.markdown('<p style="font-family:sans-serif;text-align: center; color:Black;font-size: 18px;">If you are Satisfied with the result please go ahead and Analyze</p>', unsafe_allow_html=True)
            result=st.button('Analyze',key=9)
            if result:
                hateAnalysis()
        ind=selectOptions.index('Hate Speech Analysis')
        selectOptions.pop(ind)
        addSelect()

    elif select == 'Emotion Analysis':
        with st.beta_expander("Expand Me to see the Emotion Analysis"):
            st.markdown('<p style="font-family:sans-serif;text-align: left; color:Black;font-size: 18px;">Emotion analysis is the process of identifying and analyzing the underlying emotions expressed in textual data. Emotion analytics can extract the text data from multiple sources to analyze the subjective information and understand the emotions behind it.</p>', unsafe_allow_html=True)
            st.write("Emotion Analysis uses the Hugging Face Transformer to learn more about Hugging Face 🤗 [link](https://huggingface.co/docs/transformers/main_classes/pipelines)")
            st.markdown('<p style="font-family:sans-serif;text-align: center; color:Black;font-size: 18px;">Test the Model with Example.</p>', unsafe_allow_html=True)
            
            text1 = st.text_input("",key=2)
            if text1:
                text = preprocess(text1)
                encoded_input = tokenizer(text, return_tensors='pt')
                output = model(**encoded_input)
                scores = output[0][0].detach().numpy()
                scores = softmax(scores)
                ranking = np.argsort(scores)
                ranking = ranking[::-1]
                for i in range(scores.shape[0]):
                    l = labels[ranking[i]]
                    s = scores[ranking[i]]
                    # st.write(f"{i+1}) {l} {np.round(float(s), 4)}")
                    st.write(f"{i+1}) {l} {np.round(float(s), 4)}")
            st.markdown('<p style="font-family:sans-serif;text-align: center; color:Black;font-size: 18px;">If you are Satisfied with the result please go ahead and Analyze</p>', unsafe_allow_html=True)
            
            result=st.button('Analyze',key=13)
            if result:
                emotionAnalysis()
                # random_tweet = st.radio('Shows Examples', ('amusement', 'anger', 'annoyance', 'confusion', 'disapproval', 'excitement', 'love', 'suprise'))
                # st.markdown(df.query("emotion == @random_tweet")[["text"]].sample(n=1).iat[0, 0])
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


   




    





###################################################################################




t= pivot_ui(df)

gb = GridOptionsBuilder.from_dataframe(df)
gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc='sum', editable=True)
gb.configure_side_bar()
gb.configure_pagination(paginationAutoPageSize=True)
cellsytle_jscode = JsCode("""
function(params) {
    if (params.value == 'A') {
        return {
            'color': 'white',
            'backgroundColor': 'darkred'
        }
    } else {
        return {
            'color': 'black',
            'backgroundColor': 'white'
        }
    }
};
""")
gb.configure_column("group", cellStyle=cellsytle_jscode)

with st.beta_expander("Expand Me to see the Data"):
    AgGrid(df,fit_columns_on_grid_load=True,enable_enterprise_modules=True)
# video_file = open('animation.gif', 'rb')
# video_bytes = video_file.read()

# st.video(video_bytes)

with st.beta_expander("Expand Me to see the DataFrame and Visualize"):
    with open(t.src, encoding="utf8") as t:
        components.html(t.read(), width=1300, height=1000, scrolling=True)
    

with st.beta_expander("Expand me to understand How to work with pivot table"):
    
    st.markdown("![Alt Text](https://pivottable.js.org/images/animation.gif)")





###################################################################################









