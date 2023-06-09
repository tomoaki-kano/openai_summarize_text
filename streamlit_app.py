import streamlit as st

# テキストボックスを作成
input_text = st.text_input('テキストを入力してください')

# ボタンを作成
button_clicked = st.button('表示')

# ボタンがクリックされたらテキストを表示
if button_clicked:
    st.write(input_text)