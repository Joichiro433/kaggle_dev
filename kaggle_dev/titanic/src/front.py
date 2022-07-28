import streamlit as st

st.title('Would you make it if you were on the Titanic?')

st.sidebar.header('Your Information')
sex = st.sidebar.radio('Sex', ('Male', 'Female'))
sex = 0 if sex == 'Male' else 1

age = st.sidebar.slider('How old are you?', min_value=0, max_value=100, step=1, value=25)

port = st.sidebar.radio('Port of depature', ('Queesntown, Ireland', 'Southampton, U.K.', 'Cherbourg, France'))

pclass = st.sidebar.radio('Class', (1, 2, 3))
pclass

family = st.sidebar.slider('How many family members with you?', min_value=0, max_value=10, step=1, value=0)
family

fare = st.sidebar.number_input('How much was your ticket?', min_value=0.0, max_value=500.0, step=0.1, value=15.0)
fare

execute = st.sidebar.button('Run')
if execute:
    st.text('Exexuted')
    # Pclass	Sex	Age	SibSp	Parch	Fare	Embarked	FamilySize	Title_num	TicketFreq
    [pclass, sex, ]

