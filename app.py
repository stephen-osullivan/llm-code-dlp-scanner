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

from utils import list_repo, load_readme_file, load_repo_files

## prompt template

system_prompt = """
    You are a useful AI assitant, who is an expert at reviewing code repositories. 
    Please keep your answers consise and strive to answer the users questions exactly."""
prompt = ChatPromptTemplate.from_messages(
    [
        ('system', system_prompt),
        ("human", "{question}"),
    ]
)

## model
llm = ollama.Ollama(model='llama3', stop=['<|eot_id|>'], num_ctx = 2048, temperature=0.2)
output_parser = StrOutputParser()
chain = prompt|llm|output_parser

## streamlit
st.set_page_config(page_title="Repo Exploration", layout="wide")
st.title('Repo Exploration')
repo_local_path = None
with st.sidebar:
    repo_type = st.selectbox('Repo Type', ['Local', 'Online'])
    if repo_type == 'Online':
        repo_url = st.text_input('Please enter a github repo url: https://github.com/user/repo.git')
        st.write(f'Repo: {repo_url}')
        if repo_url:
            download_repo = st.button('Download Repo')
            repo_local_path = os.path.join('repos', '/'.join(repo_url.split('/')[-2:]).split('.git')[0])
            if download_repo:
                # Clone the repository
                if os.path.exists(repo_local_path):
                    # Remove the existing directory and its contents
                    shutil.rmtree(repo_local_path)
                repo = Repo.clone_from(repo_url, repo_local_path)
    elif repo_type == 'Local':
        repo_local_path = st.text_input('Enter Repo Directory')
        st.write(f'Repo: {repo_local_path}')

if repo_local_path:
    if st.button('Load Repo Files'):  
        repo_docs = load_repo_files(repo_local_path=repo_local_path)
        prompt = """
        Please look for confidential information leaks in the below file. I'd like you to identify passwords, api_keys, or PII customer information like DOBs, addresses, and names. 
        Please only mention leaks that have you have actually discovered and do not give general suggestions on how to avoid leaks.
        Can you also summarise what the files are doing please in one sentence.
        """
        
        for doc in repo_docs:
            file_name = doc.metadata['source']
            invoke_args = {'question': prompt + '\n\n' + '#'*20 + '\n' + file_name + '#'*20 +'\n\n'+ doc.page_content}
            st.write(f'**{file_name}**:\n\n{chain.invoke(invoke_args)}')


    readme_file_string = load_readme_file(repo_local_path)

    col1, col2 = st.columns([2,1])
    with col1:
        if st.toggle('Show README.md'):
            st.markdown(readme_file_string, unsafe_allow_html=True)

    with col2:
        if st.toggle('Show Repo Files'):
            repo_list = list_repo(repo_local_path)
            repo_files = [d for d in repo_list if d['blob_type'] =='blob']
            repo_dirs = [d for d in repo_list if d['blob_type'] =='tree']
            out_string = '**Repo Files:**\n\n'
            out_string = out_string + ", ".join([f"{d['file_path']}: {d['file_size']/1024:.2f} KB" for d in repo_files])
            out_string = out_string + '\n\n**Repo Dirs:**\n\n' + '\n'.join([d['file_path'] for d in repo_dirs])
            st.write(out_string)

if repo_local_path:
    input_text = st.text_input("Please enter query")
    if input_text:
        st.write(chain.invoke({'question':input_text}))