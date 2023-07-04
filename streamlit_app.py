'''
Streamlit: 
'''

import streamlit as st

import langchain
from langchain.llms import OpenAI


def summarize_text(text):

    llm = OpenAI(
        model_name="gpt-4-0613",
        temperature=0,
        max_tokens=-1,
    )

    prompt = f'''
    以下の文章を日本語で簡潔に要約してください

    {text}
    '''

    return llm(prompt)

# テキストボックスを作成
input_text = st.text_area('要約したい文章を入力してください。')

# ボタンを作成
button_clicked = st.button('要約')

# ボタンがクリックされたらテキストを表示
if button_clicked:
    st.write(summarize_text(input_text))
