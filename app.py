"""
remember to first run $ ollama serve
"""
from git import Repo
import streamlit as st

import os
import shutil

from chains import get_chain
from utils import list_repo, load_readme_file, load_repo_files, concatenate_docs

## streamlit
st.set_page_config(page_title="Repo Exploration", layout="wide")
st.title('Repo Exploration')
repo_local_path = None
with st.sidebar:
    framework = st.selectbox('Framework', ['Ollama', 'OpenAI', 'Huggingface']).lower()
    if framework == 'ollama':
        model = st.selectbox('Model', ['llama3', 'llama2', 'deepseek-coder'])
    elif framework == 'openai':
        model = st.selectbox('Model', ['gpt-3.5-turbo'])
    elif framework == 'huggingface':
        model = st.selectbox('Model', ["mistralai/Mistral-7B-Instruct-v0.2", "google/gemma-1.1-7b-it", "meta-llama/Meta-Llama-3-8B-Instruct"])
    repo_type = st.selectbox('Repo Type', ['Local', 'Online'])
    
    if repo_type == 'Online':
        repo_url = st.text_input(
            'Please enter a github repo url: https://github.com/user/repo.git')
        st.write(f'Repo: {repo_url}')
        
        if repo_url:
            download_repo = st.button('Download Repo')
            repo_local_path = os.path.join('../temp/repos', '/'.join(repo_url.split('/')[-2:]).split('.git')[0])
            
            if download_repo:
                # Clone the repository
                if os.path.exists(repo_local_path):
                    # Remove the existing directory and its contents
                    shutil.rmtree(repo_local_path)
                repo = Repo.clone_from(repo_url, repo_local_path)
    
    elif repo_type == 'Local':
        repo_local_path = st.text_input('Enter Repo Directory e.g. ../temp/repos/user-name/repo-name')
        st.write(f'Repo: {repo_local_path}')

if repo_local_path:
    file_method = st.selectbox('File Prompt Method', ['Separate', 'Concatenated'])
    
    
    if st.button('Analyse Repo Files'):  
        repo_docs = load_repo_files(repo_local_path=repo_local_path)
        if file_method == 'Separate':
            with open('prompts/prompt-current.txt', 'r') as f:
                prompt = f.read()                           
            chain = get_chain(framework=framework, model=model, prompt=prompt)
        
            for doc in repo_docs:
                file_name = doc.metadata['source']
                file_content = doc.page_content
                if len(file_content) == 0:
                    file_content='empty file.'
                
                invoke_args = {'file_name':file_name, 'file_content':file_content}
                with st.chat_message("AI"):
                    st.write(f'{file_name}\n\n' + chain.invoke(invoke_args))
        elif file_method == 'Concatenated':
            repo_docs = load_repo_files(repo_local_path=repo_local_path)
            concatenate_docs(repo_docs, 'temp.txt')
            with open('temp.txt') as f:
                all_docs = f.read()
            with open('prompts/prompt-concatenated.txt', 'r') as f:
                prompt = f.read()                           
            chain = get_chain(framework=framework, model=model, prompt=prompt)
            with st.chat_message("AI"):
                    st.write(chain.invoke({'concatenated_files':all_docs}))

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
    with st.chat_message("Human"):
        input_text = st.text_input("Please enter query")
        if input_text:
            chain = get_chain(framework=framework, model=model)
            with st.chat_message("AI"):
                st.write(chain.invoke({'query':input_text}))