from git import Repo
import streamlit as st

import concurrent.futures
import json
import os
import shutil

from chains import get_chain
from utils import list_repo, load_readme_file, load_repo_files, concatenate_docs, extract_json, list_models

st.set_page_config(page_title="Repo Security Analysis", layout="wide")
st.title('Repo Security_Analysis')
repo_local_path = None

########################### APP FUNCTIONS #######################################

def get_multi_threaded_response(chain, docs:list, max_workers=10) -> None:
    """
    use multithreading to submitted files for analysis concurrently.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:    

        invoke_args_list = [{'file_content': doc['file_content'], 'file_name': doc['file_name']} for doc in docs]

        def get_formatted_response(chain, invoke_args):
            file_name = invoke_args['file_name']
            if len(invoke_args['file_content']) == 0:
                invoke_args['file_content'] == 'EMPTY FILE.'
            return f"{file_name}:\n\n`{extract_json(chain.invoke(invoke_args))}`"

        threads = []
        for invoke_args in invoke_args_list:
            threads.append(executor.submit(get_formatted_response, chain, invoke_args))

        # print results as and when they come in
        for future in concurrent.futures.as_completed(threads):
            with st.chat_message('AI'):
                st.write(future.result())


def select_model():
    """
    select a frame work and thena  model from those available
    """
    framework = st.selectbox('Framework', ['OpenAI', 'OpenAI-Compatible', 'Ollama', 'Huggingface']).lower()
    if framework == 'ollama':
        model = st.selectbox('Model', ['llama3', 'llama2', 'deepseek-coder'])
    elif framework == 'openai':
        model = st.selectbox('Model', ['gpt-3.5-turbo'])
    elif framework == 'openai-compatible':
        default_url =  os.environ.get('DEFAULT_ENDPOINT_URL', 'http://localhost:11434/v1') # 11434 is ollama
        endpoint_url = st.text_input('Enter endpoint URL', value = default_url)
        model = st.selectbox(list_models(endpoint_url))
    elif framework == 'huggingface':
        model = st.selectbox('Model', ["mistralai/Mistral-7B-Instruct-v0.2", "google/gemma-1.1-7b-it", "meta-llama/Meta-Llama-3-8B-Instruct"])
    return framework, model

def get_repo():
    """
    download a repo if necessary or point to a local one
    """
    repo_local_path=None
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
    
    return repo_local_path

def show_repo_files(repo_local_path):
    """
    list the files in repo
    """
    repo_list = list_repo(repo_local_path)
    repo_files = [d for d in repo_list if d['blob_type'] =='blob']
    repo_dirs = [d for d in repo_list if d['blob_type'] =='tree']
    out_string = '**Repo Files:**\n\n'
    out_string = out_string + ", ".join([f"{d['file_path']}: {d['file_size']/1024:.2f} KB" for d in repo_files])
    out_string = out_string + '\n\n**Repo Dirs:**\n\n' + '\n'.join([d['file_path'] for d in repo_dirs])
    st.write(out_string)

########################### STREAMLIT APP #######################################

### SIDE BAR
with st.sidebar:
    # side bar options
    framework, model = select_model()
    repo_local_path =  get_repo()
    
### MAIN PAGE
if repo_local_path:
    if st.button('Analyse Repo Files'):  
        # analyse the files in the repo for PII leaks.
        repo_docs = load_repo_files(repo_local_path=repo_local_path) 
        print(repo_docs)
        with open('prompts/prompt-current.txt', 'r') as f:
            prompt = f.read()                           
        chain = get_chain(framework=framework, model=model, prompt=prompt)
        get_multi_threaded_response(chain=chain, docs=repo_docs)

    col1, col2 = st.columns([2,1])
    with col1:
        if st.toggle('Show README.md'):
            readme_file_string = load_readme_file(repo_local_path)
            st.markdown(readme_file_string, unsafe_allow_html=True)

    with col2:
        if st.toggle('Show Repo Files'):
            show_repo_files(repo_local_path=repo_local_path)
