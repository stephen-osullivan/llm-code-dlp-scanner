from git import repo, Repo
import requests
import streamlit as st

import os

st.cache_data
def load_readme_file(directory):
    readme_path = os.path.join(directory, "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            readme_content = f.read()
        return readme_content
    else:
        return "No README file found in the specified directory."
    
st.cache_data
def list_repo(repo_local_path, commit_hash='HEAD', depth = 1, files_only=False):
    commit = Repo(repo_local_path).commit(commit_hash)
    tree = commit.tree
    output_list = []
    # Traverse the tree and list files with their sizes
    for blob in tree.traverse(depth=depth):
        file_path = os.path.join(repo_local_path, blob.path)
        file_size = os.path.getsize(file_path)
        if files_only and blob.type =='tree':
            continue
        output_list.append({'blob_type': blob.type, 'file_path' : blob.path, 'file_size': file_size})
    return output_list

st.cache_data
def load_repo_files(repo_local_path, depth=-1, max_size = 1024*1024*50): # 10MB
    """
    returns a list in the form: [
        {"file_name" : file_path, "file_content" : doc.page_content, "file_length" : length in chars}
    ]
    """
    from langchain_community.document_loaders import TextLoader, NotebookLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    

    splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=1)

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
                for doc in splitter.split_documents(docs):
                    documents.append(
                        {
                            "file_name" : file_path, 
                            "file_content" : doc.page_content, 
                            "file_length": len(doc.page_content)
                        })
            except Exception as e:
                print('Failed to decode:', file_path, 'Exception:', e)
    # Now you can use the 'documents' list with LangChain
    print(f"Loaded {len(documents)} documents from the changed files.")
    return documents
    
# Function to traverse directories and read file contents
def concatenate_docs(documents, output_file):

    with open(output_file, 'w') as f:
        f.flush()
        for doc in documents:
            f.writelines(["#"*50 + '\n', doc.metadata['source'], '\n' + "#"*50 + '\n\n'])
            f.writelines(doc.page_content)
            f.writelines("\n\n")


def extract_json(s):
    """
    extracts a json from a string
    """
    start = s.find('{')
    if start == -1:
        # if not a JSON then return the string.
        return s
    end = len(s) - s[::-1].find('}')    
    return s[start:end]

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
