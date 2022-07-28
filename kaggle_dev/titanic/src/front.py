"""
streamlitを用いてtitanic予測デモアプリを作成
"""

import streamlit as st
from streamlit.components.v1 import html
import numpy as np

from predict import predict_crossval


def convert_to_feature(title, sex, age, embarked, pclass, num_family, fare):
    """画面から入力された値をモデルに入力できるように整形する"""
    # convert title
    if title == 'Mr.':
        title = 0
    elif title == 'Miss.':
        title = 1
    elif title == 'Mrs.':
        title = 3
    elif title == 'Mater':
        title = 4
    # convert sex
    sex = 0 if sex == 'Male' else 1
    # convert port
    if embarked == 'Queesntown, Ireland':
        embarked = 0
    elif embarked == 'Southampton, U.K.':
        embarked = 1
    elif embarked == 'Cherbourg, France':
        embarked = 2

    family_group = 1
    ticket_group = 1
    
    X = np.array([[pclass, sex, age, fare, embarked, title, family_group, num_family, ticket_group]])
    return X

st.set_page_config(layout='wide')

# side bar
st.sidebar.header('Your Information')
title = st.sidebar.selectbox('your title', ('Mr.', 'Miss.', 'Mrs.', 'Master'))
sex = st.sidebar.radio('Sex', ('Male', 'Female'))
age = st.sidebar.slider('How old are you?', min_value=0, max_value=100, step=1, value=25)
embarked = st.sidebar.radio('Port of depature', ('Queesntown, Ireland', 'Southampton, U.K.', 'Cherbourg, France'))
pclass = st.sidebar.radio('Ticket Class', (1, 2, 3))
num_family = st.sidebar.slider('How many family members with you?', min_value=1, max_value=10, step=1, value=1)
fare = st.sidebar.number_input('How much was your ticket?', min_value=0.0, max_value=500.0, step=0.1, value=15.0)

# main page
st.title('Would you make it if you were on the Titanic?')
execute = st.sidebar.button('Run')
if execute:
    X = convert_to_feature(title, sex, age, embarked, pclass, num_family, fare)
    proba = predict_crossval(X)[0][1]  # 生存確率
    proba = proba * 100
    html('<h2 style="color:#777777;"> The survival probability is ... </h2>')
    html(f'<center><h1 style="color:#f63366; font-size: 350%;"> {proba:.2f}% </h1></center>')
