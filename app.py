"""
remember to first run $ ollama serve
"""
from git import Repo
from langchain_community.llms import ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
import shutil
import os

from utils import list_repo_files, load_readme_file

## prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ('system', "You are a useful and attentive assitant. Please respond to user queries."),
        ("human", "{question}"),
    ]
)

## model
llm = ollama.Ollama(model='llama3', stop=['<|eot_id|>'], num_ctx = 2048)
output_parser = StrOutputParser()
chain = prompt|llm|output_parser

## streamlit
st.set_page_config(page_title="Repo Exploration", layout="wide")
st.title('Repo Exploration')
repo_local_path=None
with st.sidebar:
    repo_url = st.text_input('Please enter a github repo url: https://github.com/user/repo.git')

    if repo_url:
        download_repo = st.button('Download Repo')
        repo_local_path = os.path.join('repos', '/'.join(repo_url.split('/')[-2:]).split('.git')[0])
        if download_repo:
            # Clone the repository
            if os.path.exists(repo_local_path):
                # Remove the existing directory and its contents
                shutil.rmtree(repo_local_path)
            repo = Repo.clone_from(repo_url, repo_local_path)
    
if repo_local_path:
    st.write(f'Repo: {repo_url}')
    col1, col2 = st.columns(2)
    with col1:
        if st.toggle('Show README.md'):
            st.markdown('# README File:')
            st.write(load_readme_file(repo_local_path),)
    with col2:
        if st.toggle('Show Repo Files'):
            st.markdown('# Repo File Structure')
            repo_files = list_repo_files(repo_local_path)
            print(repo_files)
            st.write('\n'.join([f"* {d['file_path']}: {d['file_size']} Bytes" for d in repo_files]))

if repo_url:
    input_text = st.text_input("Please enter query")
    if input_text:
        st.write(chain.invoke({'question':input_text}))