from git import repo, Repo
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
def load_repo_files(repo_local_path, depth=-1, max_size = 1024*1024):
    from langchain_community.document_loaders import TextLoader, NotebookLoader

    # Load the changed files into LangChain documents
    documents = []
    file_dicts = list_repo(repo_local_path, depth=depth, files_only=True)
    for file_dict in file_dicts:
        file_path, file_size = file_dict['file_path'], file_dict['file_size']
        full_path = os.path.join(repo_local_path, file_path)
        if (file_size < max_size): # less than 10kb 
            try:
                if file_path.split('.')[-1] == 'ipynb':
                    loader = NotebookLoader(full_path, include_outputs=True)
                else:
                    loader = TextLoader(full_path, autodetect_encoding=True)
                documents.extend(loader.load())
            except:
                print('Failed to decode:', file_path)
    # Now you can use the 'documents' list with LangChain
    print(f"Loaded {len(documents)} documents from the changed files.")
    doc_names = [doc.metadata['source'] for doc in documents]
    print('Lengths:', [(n, len(doc.page_content)) for n, doc in zip(doc_names, documents)])
    return documents
    
# Function to traverse directories and read file contents
def concatenate_docs(documents, output_file):

    with open(output_file, 'w') as f:
        f.flush()
        for doc in documents:
            f.writelines(["#"*50 + '\n', doc.metadata['source'], '\n' + "#"*50 + '\n\n'])
            f.writelines(doc.page_content)
            f.writelines("\n\n")