import streamlit as st
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
from liwc import Liwc
parse, category_names = Liwc.load_token_parser('mfd2.0.dic')