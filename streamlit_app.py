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

    prompt_template = """Write a concise summary of the following:


    {text}


    CONCISE SUMMARY IN JAPANESE:"""

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
    refine_template = (
        "Your job is to produce a final summary\n"
        "We have provided an existing summary up to a certain point: {existing_answer}\n"
        "We have the opportunity to refine the existing summary"
        "(only if needed) with some more context below.\n"
        "------------\n"
        "{text}\n"
        "------------\n"
        "Given the new context, refine the original summary in Japanese"
        "If the context isn't useful, return the original summary."
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
input_text = st.text_input('要約したい文章を入力してください。')

# ボタンを作成
button_clicked = st.button('表示')


# ボタンがクリックされたらテキストを表示
if button_clicked:
    st.write(summarize_text(input_text))
