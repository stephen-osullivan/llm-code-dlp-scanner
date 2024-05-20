from git import Repo
import streamlit as st

import concurrent.futures
from io import StringIO
import json
import os
import shutil

from chains import get_chain
from utils import list_repo, load_readme_file, load_repo_files, concatenate_docs, extract_json, list_models

st.set_page_config(page_title="Repo Security Analysis", layout="wide")
st.title('Repo Security Analysis 🪬')

REPO_SAVE_DIR = os.environ.get('REPO_SAVE_DIR', 'temp/repos')
repo_local_path = None

########################### APP FUNCTIONS #######################################
def get_multi_threaded_response(chain, docs:list, max_workers=10) -> None:
    """
    use multithreading to submitted files for analysis concurrently.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:    

        def get_formatted_response(chain, doc):
            invoke_args = {k: doc[k] for k in ['file_name', 'file_content']}
            file_name = invoke_args['file_name']
            if len(invoke_args['file_content']) == 0:
                invoke_args['file_content'] == 'EMPTY FILE.'
            #return invoke_args['file_name'], extract_json(chain.invoke(invoke_args))
            return doc, chain.invoke(invoke_args)

        threads = []
        for doc in docs:
            threads.append(executor.submit(get_formatted_response, chain, doc))

        # print results as and when they come in
        for future in concurrent.futures.as_completed(threads):
            with st.chat_message('AI'):
                doc, response = future.result()
                st.write(doc['file_name'])
                st.write(f"Chunk {doc['chunk_idx']+1} of {doc['total_chunks']}")
                st.write()
                try:
                    #response = json.loads(response)
                    st.json(response)
                except:
                    st.write(response)

def select_model():
    """
    select a frame work and thena  model from those available
    """
    endpoint_url=None
    framework = st.selectbox('Framework', ['OpenAI-Compatible', 'OpenAI', 'Ollama', 'Huggingface', 'vLLM']).lower()
    
    if framework == 'ollama':
        model = st.selectbox('Model', ['llama3', 'llama2', 'deepseek-coder'])
    
    elif framework == 'openai':
        model = st.selectbox('Model', ['gpt-3.5-turbo'])
    
    elif framework == 'openai-compatible':
        default_url =  os.environ.get('DEFAULT_ENDPOINT_URL', 'http://localhost:11434/v1') # 11434 is ollama
        endpoint_url = st.text_input('Enter endpoint URL', value = default_url)
        model = st.selectbox('Model', list_models(endpoint_url))
    
    elif framework == "vllm":
        default_url =  os.environ.get('DEFAULT_ENDPOINT_URL', 'http://localhost:8000/v1') # 8000 is vllm
        endpoint_url = st.text_input('Enter endpoint URL', value = default_url)
        model = st.text_input('Model', "TheBloke/Mistral-7B-Instruct-v0.2-AWQ")

    elif framework == 'huggingface':
        model = st.selectbox('Model', ["mistralai/Mistral-7B-Instruct-v0.2", "google/gemma-1.1-7b-it", "meta-llama/Meta-Llama-3-8B-Instruct"])
    
    return framework, model, endpoint_url

def get_repo():
    """
    download a repo if necessary or point to a local one
    """
    repo_local_path=None
    repo_type = st.selectbox('Repo Type', ['Local', 'Online'])
    
    if repo_type == 'Online':
        repo_url = st.text_input(
            'Please enter a github repo url: https://github.com/user/repo.git')
        
        if repo_url:
            st.write(f'**Repo:** {repo_url}')
            download_repo = st.button('Download Repo')
            repo_local_path = os.path.join(REPO_SAVE_DIR, '/'.join(repo_url.split('/')[-2:]).split('.git')[0])
            
            if download_repo:
                # Clone the repository
                if os.path.exists(repo_local_path):
                    # Remove the existing directory and its contents
                    shutil.rmtree(repo_local_path)
                repo = Repo.clone_from(repo_url, repo_local_path)
                st.write('Download Successful.')
    
    elif repo_type == 'Local':
        repo_local_path = st.text_input('Enter Repo Path')
        if repo_local_path:
            st.write(f'**Repo**: {repo_local_path}')
    
    return repo_local_path

@st.cache_data
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

def get_current_prompt() -> str:
    with open('prompts/prompt-current.txt', 'r') as f:
        prompt = f.read()
    return prompt

########################### STREAMLIT APP #######################################

### SIDE BAR
with st.sidebar:
    # side bar options
    st.write('**Select Model:**')
    framework, model, endpoint_url = select_model()
    st.write('#')
    st.write('**Select Repo:**')
    repo_local_path =  get_repo()

### Main section
if repo_local_path and os.path.isdir(repo_local_path):
    # Display Repo Metrics in header
    files = list_repo(repo_local_path, depth=-1, files_only=True)
    num_files = len(files)
    size = sum(d['file_size'] for d in files)
    if size < (1024*1024*1024):
        size =  f'{size/(1024*1024):.2f} MB'
    else:
        size =  f'{size/(1024*1024*1024):.2f} GB'

    col1, col2 = st.columns((1,2))
    col1.metric('Files in Repo', num_files)
    col2.metric('Repo Size', size)

tab1, tab2, tab3, tab4 = st.tabs(['View Repo', 'Scan Repo', 'Change Prompt', 'Clear Cache'])
with tab1:
    # View Repo Readme or Files
    if repo_local_path:
        col1, col2 = st.columns([2,1])
        with col1:
            if st.toggle('Show README.md'):
                readme_file_string = load_readme_file(repo_local_path)
                st.markdown(readme_file_string, unsafe_allow_html=True)

        with col2:
            if st.toggle('Show Repo Files'):
                show_repo_files(repo_local_path=repo_local_path)
    else:
        st.write('**Use sidebar to select a repo**')

with tab2:
    # Analyse Repo files with LLM
    if repo_local_path:
        if st.button('Analyse Repo Files'):  
            # analyse the files in the repo for PII leaks.
            docs = load_repo_files(repo_local_path=repo_local_path) 
            prompt = get_current_prompt()                        
            chain = get_chain(framework=framework, model=model, prompt=prompt, endpoint_url= endpoint_url)
            get_multi_threaded_response(chain=chain, docs=docs)
    else:
        st.write('**Use sidebar to select a repo**')

with tab3:
    ### Options to view and change Prompt
    col1, col2 = st.columns((1,1))
    
    with col1:
        if st.button('Reset Prompt to Default'):
            shutil.copyfile('prompts/prompt-default.txt', 'prompts/prompt-current.txt')
        st.write('**Current Prompt**')
        current_prompt_textbox = st.text(get_current_prompt())

    with col2:
        change_prompt_button = st.button('Change Prompt')
        new_prompt = st.text_area('**Input New Prompt**', height = 600)
        if change_prompt_button and new_prompt != "":
            with open('prompts/prompt-current.txt', 'w') as f:
                f.write(new_prompt)
            current_prompt_textbox.text(get_current_prompt())
        st.write('**Prompt must have arguments {file_name} and {file_content} and should return a JSON**')
    
with tab4:
    st.write('**Repos Downloaded:**')
    repos = []
    users = os.listdir(REPO_SAVE_DIR)
    for user in users:
        user_repos = os.listdir(os.path.join(REPO_SAVE_DIR, user))
        repos.extend([os.path.join(user, r) for r in user_repos])
    st.write(repos)
    if st.button('Clear Downloaded Repos'):
        for r in repos:
            shutil.rmtree(os.path.join(REPO_SAVE_DIR, r))