'''
# Installation tensorflow + transformers + pipelines
# You need this to summarize the SERP and to run question-answering on the extracted corpus of text

!pip install transformers --upgrade # details https://github.com/huggingface/transformers/releases/tag/v2.6.0
!pip install tensorflow==2.1 # you need this to use T5'''

import streamlit as st
import re
import pandas as pd
import numpy as np
import trafilatura
import pprint
from googlesearch import search
import ktrain
from ktrain import text, predictor

st.title('Schema-Classifier web-app')

uQuery = st.text_input('Search what you want to classify')

uNum = st.slider('Select the number of results you want')

predictor = ktrain.load_predictor('./tmp/schema_mapping')

def getResults(uQuery, uTLD, uNum, uStart, uStop):
    query = uQuery

    d = []

    for j in search(query, tld=uTLD, num=uNum, start=uStart, stop=uStop, pause=2):
        d.append(j)
        print(j)
    return d


results = getResults(uQuery, "com", uNum, 1, uNum)

pd.set_option('display.max_colwidth', None)  # make sure output is not truncated (cols width)
pd.set_option("display.max_rows", 100)  # make sure output is not truncated (rows)


def readResults(urls, query):
    # Prepare the data frame to store results
    x = []
    position = 0  # position on the serp

    # Loop items in results
    for page in urls:
        position += 1
        downloaded = trafilatura.fetch_url(page)
        if downloaded is not None:  # assuming the download was successful
            result = trafilatura.extract(downloaded, include_tables=False, include_formatting=False,
                                         include_comments=False)
            label = pred(result, predictor)
            x.append((page, result, query, position, label))
    return x


def pred(text, predictor):
    return predictor.predict(text)

if uQuery and uNum is not None:
    d = readResults(results, uQuery)  # get results from the query

    df = pd.DataFrame(d, columns=('url', 'result', 'query', 'position', 'type'))  # store data in a data frame

    print("total number of articles (before filtering) ", len(df))

    # Remove rows where result is empty
    df['result'].replace(' ', np.nan, inplace=True)
    df_final = df.dropna(subset=['result'])

    # Remove rows where article are less than 200 characters in lenght
    df_final = df_final[df_final['result'].apply(lambda x: len(str(x)) > 200)]

    # Reindex df
    #df_final.index = range(len(df_final.index))

    # Set the file name
    # cleanQuery = re.sub('\W+', '', uQuery)
    # file_name = cleanQuery + ".csv"

    # Store data to CSV
    # df_final.to_csv(file_name, encoding='utf-8', index=True)
    # print("total number of articles saved on", file_name, len(df_final))


    # getting text ready by merging all pages together (no index)
    # full_body = df_final[['result']].agg(''.join, axis=1).to_string(index=False).strip()

    st.dataframe(df)
