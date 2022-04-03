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


from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax


# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)
# MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
# tokenizer = AutoTokenizer.from_pretrained(MODEL)
# config = AutoConfig.from_pretrained(MODEL)
# # PT
# model = AutoModelForSequenceClassification.from_pretrained(MODEL)
# #model.save_pretrained(MODEL)

# text = st.text_input("Some input")

# if text:
#     text = preprocess(text)
#     st.write(text)
#     encoded_input = tokenizer(text, return_tensors='pt')
#     output = model(**encoded_input)
#     st.write(output)
#     scores = output[0][0].detach().numpy()
#     st.write(scores)
#     scores = softmax(scores)
#     st.write(scores)
#     # # TF
#     # model = TFAutoModelForSequenceClassification.from_pretrained(MODEL)
#     # model.save_pretrained(MODEL)
#     # text = "Covid cases are increasing fast!"
#     # encoded_input = tokenizer(text, return_tensors='tf')
#     # output = model(encoded_input)
#     # scores = output[0][0].numpy()
#     # scores = softmax(scores)
#     # Print labels and scores
#     ranking = np.argsort(scores)
#     ranking = ranking[::-1]
#     for i in range(scores.shape[0]):
#         l = config.id2label[ranking[i]]
#         st.write(l)
#         s = scores[ranking[i]]
#         st.write(f"{i+1}) {l} {np.round(float(s), 4)}")







import json



from transformers import RobertaTokenizerFast, TFRobertaForSequenceClassification, pipeline

tokenizer1 = RobertaTokenizerFast.from_pretrained("arpanghoshal/EmoRoBERTa")
model = TFRobertaForSequenceClassification.from_pretrained("arpanghoshal/EmoRoBERTa")
emotion = pipeline('sentiment-analysis', 
                    model='arpanghoshal/EmoRoBERTa')

text1 = st.text_input("Test Our Model with Example",key=2)
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
    

# st.write(listdict.values())



# emotion_labels = emotion("Thanks for using it.")
# st.write(emotion_labels)