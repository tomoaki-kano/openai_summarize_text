import streamlit as st

import langchain
from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document


def summarize_text(text):

    llm = OpenAI(temperature=0)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2500
    )

    texts = text_splitter.split_text(text)

    docs = [Document(page_content=t) for t in texts]

    prompt_template = """下記の文章を簡潔に日本語で要約してください。:


    {text}


    簡潔な要約:"""

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
    refine_template = (
        "あなたの役割は最終的な要約文を作成することです\n"
        "私はすでに要約したポイントを提示します: {existing_answer}\n"
        "すでに存在する要約と合わせて、新しい要約を生成してください"
        "(もし必要であれば) 以下の文章を要約文に含めてください\n"
        "------------\n"
        "{text}\n"
        "------------\n"
        "新しい文脈を提供するので, 元々の要約文を再生成してください"
        "もし提供された新しい文脈が不要な場合は, 元々の要約文を回答してください"
    )
    refine_prompt = PromptTemplate(
        input_variables=["existing_answer", "text"],
        template=refine_template,
    )
    chain = load_summarize_chain(OpenAI(temperature=0), chain_type="refine", return_intermediate_steps=True, question_prompt=PROMPT, refine_prompt=refine_prompt)
    chain({"input_documents": docs}, return_only_outputs=True)

    chain = load_summarize_chain(llm, chain_type="refine")

    return chain.run(docs)


# テキストボックスを作成
input_text = st.text_area('要約したい文章を入力してください。')

prompt_template = st.text_area(
    "prompt_template",
    value = """下記の文章を簡潔に日本語で要約してください。:


{text}


簡潔な要約:"""
)

refine_template = st.text_area(
    "refine_template",
    value = """あなたの役割は最終的な要約文を作成することです
私はすでに要約したポイントを提示します: {existing_answer}
すでに存在する要約と合わせて、新しい要約を生成してください
(もし必要であれば) 以下の文章を要約文に含めてください
------------
{text}
------------
新しい文脈を提供するので, 元々の要約文を再生成してください
もし提供された新しい文脈が不要な場合は, 元々の要約文を回答してください"""
)

# ボタンを作成
button_clicked = st.button('要約')


# ボタンがクリックされたらテキストを表示
if button_clicked:
    st.write(summarize_text(input_text))
