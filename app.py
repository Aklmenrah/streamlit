import pprint
import re

import ktrain
import numpy as np
import pandas as pd
import streamlit as st
import trafilatura
from googlesearch import search
from ktrain import text, predictor
#import eli5

st.title(':crystal_ball: schemaPredictor :crystal_ball:')

add_selectbox = st.selectbox(
    'How would you like to predict?',
    ('Text', 'Url'))

predictor = ktrain.load_predictor('./tmp/schema_mapping')


def get_prob(p):
    i = 0
    for x in p:
        if x > i:
            i = x
    return i


if add_selectbox == "Text":
    body = st.text_area('Insert your text here, as clean as possible.')
    if st.button("Predict"):
        st.success(":crystal_ball: " + predictor.predict(body) + " :crystal_ball:")
        st.success("With a probability of " + "{:.1%}".format(get_prob(predictor.predict_proba(body))))
        #st.write("How did we get that result:")
        #st.success(eli5.formatters.text.format_as_text(eli5.explain_prediction(predictor, body)))


elif add_selectbox == "Url":
    body = st.text_input('Insert your url here')
    if st.button("Predict"):
        page = body
        downloaded = trafilatura.fetch_url(page)
        result = trafilatura.extract(downloaded, include_tables=False, include_formatting=False,
                                     include_comments=False)
        st.success(":crystal_ball: " + predictor.predict(result) + " :crystal_ball:")
        st.success("With a probability of " + "{:.1%}".format(get_prob(predictor.predict_proba(result))))
        #st.write("How did we get that result:")
        #st.success(eli5.formatters.text.format_as_text(eli5.explain_prediction(predictor, result)))


