from git import repo, Repo
from git.refs import RemoteReference
import pandas as pd
import requests
import streamlit as st


import os
import shutil

# ENVIROMENT VARIABLES
DOC_CHUNK_CHARS = int(os.environ.get('DOC_CHUNK_CHARS', 1_000)) # 1_000 characters chunks
MAX_FILE_SIZE = int(os.environ.get('MAX_FILE_SIZE', 20*1024*1024)) # 20 MB
MAX_DOC_CHARS = int(os.environ.get('MAX_DOC_CHARS', 15_000)) # 15_000 characters max doc size before chunking
REPO_SAVE_DIR = os.environ.get('REPO_SAVE_DIR', 'temp/repos')

def download_git_repo(url: str) -> str:
    """
    download a repo using git python and saves it in REPO_SAVE_DIR
    """

    repo_local_path = os.path.join(REPO_SAVE_DIR, '/'.join(url.split('/')[-2:]).split('.git')[0])
            
    if os.path.exists(repo_local_path):
        # Remove the existing directory and its contents
        shutil.rmtree(repo_local_path)

    # Clone the repository
    Repo.clone_from(url, repo_local_path)
    return repo_local_path

def list_branches(repo_local_path:str) -> list:
    """
    takes a git repo path and returns a list of branches 
    """
    repo = Repo(repo_local_path)
    return [ref.name.removeprefix('origin/') for ref in repo.refs if isinstance(ref, RemoteReference)]

def switch_branch(local_repo_path:str, branch_name:str):
    """
    takes a local repo path and branch name and then checks out that branch
    """
    branches = list_branches(local_repo_path)
    if branch_name in branches:
        repo = Repo(local_repo_path)
        repo.git.checkout(branch_name)
    else:
        raise ValueError('Branch does not exist')

st.cache_data
def list_repo(repo_local_path, depth = 1, files_only=False) -> list:
    """
    go through a repo a list the files and their size
    """
    commit = Repo(repo_local_path).commit('HEAD')
    tree = commit.tree
    output_list = []
    # Traverse the tree and list files with their sizes
    for blob in tree.traverse(depth=depth):
        file_path = os.path.join(repo_local_path, blob.path)
        if os.path.isfile(file_path):
            file_size = os.path.getsize(file_path)
            if files_only and blob.type =='tree':
                continue
            output_list.append({'blob_type': blob.type, 'file_path' : blob.path, 'file_size': file_size})
    return output_list

st.cache_data
def load_repo_files(repo_local_path, depth=-1, max_size = MAX_FILE_SIZE):
    """
    returns a list in the form: [
        {"file_name" : file_path, "file_content" : doc.page_content, "file_length" : length in chars}
    ]
    """
    from langchain_community.document_loaders import TextLoader, NotebookLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    

    splitter = RecursiveCharacterTextSplitter(chunk_size=DOC_CHUNK_CHARS, chunk_overlap=1)

    # Load the changed files into LangChain documents
    documents = []
    file_dicts = list_repo(repo_local_path, depth=depth, files_only=True)
    for file_dict in file_dicts:
        file_path, file_size = file_dict['file_path'], file_dict['file_size']
        full_path = os.path.join(repo_local_path, file_path)
        if (file_size < max_size): # less than 10MB
            try:
                if file_path.split('.')[-1] == 'ipynb':
                    loader = NotebookLoader(full_path, include_outputs=True)
                else:
                    loader = TextLoader(full_path, autodetect_encoding=True)
                
                docs = loader.load()
                full_doc_len = len(docs[0].page_content)
                if full_doc_len < MAX_DOC_CHARS:
                    split_docs = splitter.split_documents(docs)
                    start_line = 0
                    for chunk_idx, doc in enumerate(split_docs):
                        documents.append(
                            {   
                                "file_name" : file_path, 
                                "file_content" : doc.page_content, 
                                "file_length": len(doc.page_content),
                                "chunk_idx": chunk_idx,
                                "total_chunks": len(split_docs),
                                "start_line" : start_line,
                            })
                        start_line += doc.page_content.count('\n') + 1
            except Exception as e:
                print('Failed to decode:', file_path, 'Exception:', e)
    # Now you can use the 'documents' list with LangChain
    print(f"Loaded {len(documents)} documents from the changed files.")
    
    return documents

st.cache_data
def load_readme_file(directory):
    """
    read the readme filde in a directory
    """
    readme_path = os.path.join(directory, "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            readme_content = f.read()
        return readme_content
    else:
        return "No README file found in the specified directory."
    

    
def list_models(endpoint_url):
    """
    list the models available at the endpoint
    """
    API_KEY = os.environ.get('OPENAI_API_KEY', 'dummytoken')
    
    try:
        r = requests.get(os.path.join(endpoint_url, 'models'), headers={"Authorization": f"Bearer {API_KEY}"})
        if r.status_code == 200:
            return [d['id'] for d in r.json()['data']]
        else:
            return [f'failed to connect. Status code: {r.status_code}']
    except Exception as e:
        print(e)
        return ['Failed to connect to host.']

@st.cache_data
def responses_to_df(response_list):
    # converts the list of responses into a dataframe
    df = pd.DataFrame([r['response'] for r in response_list])
    df['chunk_idx'] = [r['chunk_idx'] for r in response_list]
    df['file name'] = [r['file_name'] for r in response_list]
    df['start_line'] = [r['start_line'] for r in response_list]
    return df

@st.cache_data
def summarise_responses(responses_df):
    # summarise the responses data frame to show the leaks in each fiel
    df_summary = responses_df.groupby('file name').agg({'file description': 'first', 'sensitive data count': 'sum'})
    return df_summary

@st.cache_data
def get_leaks_df(responses_df):
    # takes the dataframe of responses and unpacks the leaks into a new dataframe
    df_leaks = []
    responses_df = responses_df[responses_df['sensitive data'].apply(len)>0]
    for idx, row in  responses_df.iterrows():
        for d in row['sensitive data']:
            new_row = {'file name' : row['file name']} | d
            new_row['line_number'] += row['start_line']
            df_leaks.append(new_row)
    return pd.DataFrame(df_leaks)

def batch_load(iterator, batch_size):
    """
    batch loads an iterator in python
    """
    start, end = 0, batch_size
    
    while start < len(iterator):
        yield iterator[start:end]
        start = end
        end += batch_size
