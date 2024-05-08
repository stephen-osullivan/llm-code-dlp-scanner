from git import repo, Repo

import os

def load_readme_file(directory):
    readme_path = os.path.join(directory, "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            readme_content = f.read()
        return readme_content
    else:
        return "No README file found in the specified directory."

def list_repo_files(repo_local_path, commit_hash='HEAD'):
    commit = Repo(repo_local_path).commit(commit_hash)
    tree = commit.tree
    output_list = []
    # Traverse the tree and list files with their sizes
    for blob in tree.traverse():
        if blob.type == "blob":
            file_path = os.path.join(repo_local_path, blob.path)
            file_size = os.path.getsize(file_path)
            output_list.append({'file_path' : file_path, 'file_size': file_size})
    return output_list