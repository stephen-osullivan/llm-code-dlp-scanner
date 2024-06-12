from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from git import Repo
import pandas as pd
import streamlit as st

import concurrent.futures
from io import StringIO
import json
import math
import os
import shutil

from chains import get_chain
from utils import (
    download_git_repo, list_branches, switch_branch, list_repo, load_readme_file, load_repo_files, 
    list_models, responses_to_df, summarise_responses, get_leaks_df, batch_load
)
from output_pydantic import ResponseOutput

st.set_page_config(page_title="Repo Leak Scanner", layout="wide")
st.title('Repo Leak Scanner 🪬')

REPO_SAVE_DIR = os.environ.get('REPO_SAVE_DIR', 'temp/repos')
CONCURRENT_REQUEST_LIMIT = os.environ.get('CONCURRENT_REQUEST_LIMIT', 200)
VALIDATE_RESPONSES = True
DEBUG_RESPONSES = True

########################### APP FUNCTIONS #######################################
def app_get_multi_threaded_response(chain, docs:list, max_workers=10) -> None:
    """
    use multithreading to submit files for analysis concurrently.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:    

        def get_formatted_response(chain, doc):
            invoke_args = {k: doc[k] for k in ['file_name', 'file_content']}
            file_name = invoke_args['file_name']
            if len(invoke_args['file_content']) == 0:
                invoke_args['file_content'] == 'EMPTY FILE.'
            # replace "" in file with ' ' to allow json formatting
            invoke_args['file_content'] = invoke_args['file_content'].replace("\"","'")
            response = chain.invoke(invoke_args) 
            response = StrOutputParser().invoke(response).replace("\\", "") # python can't handle these when json decoding
            return doc, response
        
        num_batches = math.ceil(len(docs)/CONCURRENT_REQUEST_LIMIT)
        st.session_state["response_list"] = list()
        batch_progress_bar = st.progress(0, f'Running batches')
        response_progress_bar = st.progress(0, text = f'Sending Requests')
        for batch_idx, batch in enumerate(batch_load(docs, batch_size=CONCURRENT_REQUEST_LIMIT)):
            batch_progress_bar.progress(batch_idx/num_batches, f'Running batch {batch_idx+1} of {num_batches}')
            batch_size = len(batch)
            threads = []
            
            response_progress_bar.progress(0, text = f'Sending Requests')
            for idx, doc in enumerate(batch):
                # submit threads
                threads.append(executor.submit(get_formatted_response, chain, doc))

            # print results as and when they come in            
            for idx, future in enumerate(concurrent.futures.as_completed(threads)):
                response_progress_bar.progress((idx+1)/batch_size, text = f'Completed request {idx+1} of {batch_size}' )
                json_parser, str_parser = JsonOutputParser(pydantic_object=ResponseOutput), StrOutputParser()
                with st.chat_message('AI'):
                    doc, response = future.result()
                    file_name, chunk_idx, start_line = doc['file_name'], doc['chunk_idx'], doc['start_line']
                    total_chunks = doc['total_chunks']
                    st.write(file_name, f"Chunk {chunk_idx+1} of {total_chunks}")
                    st.write()
                    try:
                        # try to convert to json
                        response = json_parser.invoke(response)
                        if 'sensitive_data_list' not in response:
                            response['sensitive_data_list'] = []
                        if 'sensitive_data_count' not in response:
                            response['sensitive_data_count'] = 0
                        # check for hallucination
                        sensitive_data_list = response['sensitive_data_list']
                        for idx, data_element in enumerate(sensitive_data_list):
                            if data_element['sensitive_data'] not in doc['file_content'].replace("\"","'"):
                                data_element['in_file'] = False
                            else:
                                data_element['in_file'] = True

                        st.json(response)
                        st.session_state["response_list"].append(
                            dict(file_name = file_name, 
                                 chunk_idx = chunk_idx,
                                 total_chunks = total_chunks,
                                 start_line = start_line, 
                                 response = response))
                        
                    except BaseException as e:
                        if DEBUG_RESPONSES:
                            st.write(e)
                        try:
                            st.write(str_parser.invoke(response))
                        except Exception:
                            st.write(response)
        batch_progress_bar.progress(100, f'Finished {num_batches} batches.')

def validate_responses(validation_chain):
    """
    loop through response and validate them one by one with a second llm call
    """
    validation_progress_bar = st.progress(0, f'Validating Responses')
    def validate_response(file_name, data, response_idx, data_idx,):
        validation = validation_chain.invoke({'data': data})
        return response_idx, data_idx, validation
    
    response_list = st.session_state['response_list']
    n = len(response_list)
    if n > 0:
        threads = []
        n_threads = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:   
            for response_idx, response in enumerate(response_list):
                file_name = response['file_name']
                response = response['response']
                for data_idx, data in enumerate(response['sensitive_data_list']):
                    try:
                        data =  data['description'] + " : " + data['sensitive_data']
                        threads.append(
                        executor.submit(
                            validate_response, 
                            file_name,
                            data,
                            response_idx, 
                            data_idx))    
                   
                        n_threads += 1
                    except Exception as e:
                        pass
            for idx, future in enumerate(concurrent.futures.as_completed(threads)):
                response_idx, data_idx, validation = future.result()
                response_list[response_idx]['response']['sensitive_data_list'][data_idx]['is_leak'] = validation
                validation_progress_bar.progress((idx+1)/n_threads, text = f'Completed validation {idx+1} of {n_threads}')

def app_select_model():
    """
    select a frame work and thena  model from those available
    """
    endpoint_url=None
    framework = st.selectbox('Framework', ['vLLM', 'OpenAI-Compatible', 'OpenAI', 'Ollama', 'Huggingface']).lower()
    
    if framework == 'ollama':
        model = st.selectbox('Model', ['mistral', 'llama3', 'llama2', 'deepseek-coder'])
    
    elif framework == 'openai':
        model = st.selectbox('Model', ['gpt-3.5-turbo'])
    
    elif framework == 'openai-compatible':
        default_url =  os.environ.get('DEFAULT_ENDPOINT_URL', 'http://localhost:11434/v1') # 11434 is ollama
        endpoint_url = st.text_input('Enter endpoint URL', value = default_url)
        model = st.selectbox('Model', list_models(endpoint_url))
    
    elif framework == "vllm":
        default_url =  os.environ.get('DEFAULT_ENDPOINT_URL', 'http://localhost:8000/v1') # 8000 is vllm
        endpoint_url = st.text_input('Enter endpoint URL', value = default_url)
        model = st.selectbox('Model', list_models(endpoint_url))

    elif framework == 'huggingface':
        model = st.selectbox('Model', ["mistralai/Mistral-7B-Instruct-v0.2", "google/gemma-1.1-7b-it", "meta-llama/Meta-Llama-3-8B-Instruct"])
    
    return framework, model, endpoint_url

def app_get_repo():
    """
    download a repo if necessary or point to a local one
    """
    repo_type = st.selectbox('Repo Type', ['Local', 'Online'])
    if repo_type == 'Online':
        repo_url = st.text_input('Please enter a github repo url: https://github.com/user/repo.git')
        if repo_url:
            download_repo = st.button('Download Repo')
            if download_repo:
                local_repo_path = download_git_repo(repo_url)
                st.toast('Download Successful.', icon='✅')
                st.session_state['local_repo_path'] = local_repo_path
                st.session_state['docs']=None
    
    elif repo_type == 'Local':
        local_repo_path = st.text_input('Enter Repo Path')
        if local_repo_path:
            if os.path.exists(local_repo_path):
                st.toast('Local Repo found.', icon='✅')
                st.session_state['local_repo_path'] = local_repo_path
    

@st.cache_data
def app_show_repo_files(repo_local_path):
    """
    list the files in repo
    """
    repo_list = list_repo(repo_local_path, depth=-1)
    total_size = sum(blob['file_size'] for blob in repo_list)
    for blob in repo_list:
        # format responses
        if blob['blob_type'] == 'blob':
            blob['blob_type'] = 'file'
        elif blob['blob_type'] == 'tree':
            blob['blob_type'] = 'directory'
        blob['file_size'] = blob['file_size']/(1024) # MB

    st.dataframe(
        pd.DataFrame(repo_list)[['file_path', 'file_size']],
        column_config={
            'file_path' : 'file',
            'file_size' : st.column_config.NumberColumn('size', format="%d KB", disabled=True)},
        use_container_width=True)

def get_current_prompt() -> str:
    with open('prompts/prompt-current.txt', 'r') as f:
        prompt = f.read()
    return prompt

########################### STREAMLIT APP #######################################

# Initialise Session State
session_state_vars = ['local_repo_path', 'docs', 'response_list']
for key in session_state_vars:
    if key not in st.session_state:
        st.session_state[key] = None

### SIDE BAR
with st.sidebar:
    # side bar options
    framework, model, endpoint_url = app_select_model()
    app_get_repo()
    if st.session_state['local_repo_path']:
        branch_name = st.selectbox('Switch Branch', list_branches(st.session_state['local_repo_path']))
        if st.button('Switch'):
            switch_branch(st.session_state['local_repo_path'], branch_name)
        
### Main section

if st.session_state['local_repo_path']:
    # Display Repo Metrics in header
    files = list_repo(st.session_state['local_repo_path'], depth=-1, files_only=True)
    num_files = len(files)
    size = sum(d['file_size'] for d in files)
    if size < (1024*1024*1024):
        size =  f'{size/(1024*1024):.2f} MB'
    else:
        size =  f'{size/(1024*1024*1024):.2f} GB'

    col1, col2 = st.columns((1,2))
    col1.metric('Files in Repo', num_files)
    col2.metric('Repo Size', size)

tab1, tab2, tab3, tab4, tab5 = st.tabs(['View Repo', 'Scan Repo', 'Scan Results', 'Change Prompt', 'Clear Cache'])
with tab1:
    # View Repo Readme or Files
    if st.session_state['local_repo_path']:
        col1, col2 = st.columns([2,1])
        with col1:
            if st.toggle('Show README.md'):
                readme_file_string = load_readme_file(st.session_state['local_repo_path'])
                st.markdown(readme_file_string, unsafe_allow_html=True)

        with col2:
            if st.toggle('Show Repo Files'):
                app_show_repo_files(repo_local_path=st.session_state['local_repo_path'])
    else:
        st.write('**Use sidebar to select a repo**')

with tab2:
    # Analyse Repo files with LLM
    if st.session_state['local_repo_path']:
        if st.button('Load Repo Files'): 
            st.session_state['response_list'] = None
            # analyse the files in the repo for PII leaks.
            st.session_state['docs'] = load_repo_files(repo_local_path=st.session_state['local_repo_path'])     

        if st.session_state['docs']:
            num_docs = len(st.session_state['docs'])
            num_batches = math.ceil(num_docs/CONCURRENT_REQUEST_LIMIT)
            st.write(f"**{num_docs}** requests will be made. **{num_batches}** batch{'es' if num_batches>1 else''}")
            if st.button('Initiate Scan'):
                prompt = get_current_prompt()                        
                chain = get_chain(framework=framework, model=model, prompt=prompt, endpoint_url= endpoint_url,
                                  parser = JsonOutputParser(pydantic_object=ResponseOutput))
                app_get_multi_threaded_response(chain=chain, docs=st.session_state['docs'])
                with open('prompts/prompt-validation.txt') as f:
                    prompt_validation = f.read()  
                validation_chain = get_chain(framework=framework, model=model, prompt=prompt_validation, endpoint_url=endpoint_url) | StrOutputParser()
                if VALIDATE_RESPONSES:
                    validate_responses(validation_chain=validation_chain)

    else:
        st.write('**Use sidebar to select a repo**')

with tab3:
    # summarise app responses into a pair of dataframes
    if st.session_state['response_list']:
        responses_df = responses_to_df(st.session_state['response_list'])
        summary_df = summarise_responses(responses_df)
        leaks_df = get_leaks_df(responses_df)
        st.metric('Possible Leaks Found', summary_df['sensitive_data_count'].sum())
        if st.toggle('Show Summary', True):
            st.dataframe(
                summary_df, use_container_width=True,
                column_config = {
                "sensitive_data_count" : st.column_config.ProgressColumn(
                    "sensitive_data_count", min_value = 0, max_value = 10,
                    format="%i")})
            
        if st.toggle('Show Possible Leaks', True):
            st.dataframe(leaks_df, use_container_width=True)

with tab4:
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
    
with tab5:
    repos = []
    users = os.listdir(REPO_SAVE_DIR)
    for user in users:
        # loop over all users in the folder
        user_path = os.path.join(REPO_SAVE_DIR, user)
        if os.path.isdir(user_path):
            user_repos = os.listdir(user_path)
            repos.extend([os.path.join(user, r) for r in user_repos if os.path.isdir(os.path.join(user_path, r))])
    st.write(repos)
    if st.button('Clear Downloaded Repos'):
        for r in repos:
            shutil.rmtree(os.path.join(REPO_SAVE_DIR, r))