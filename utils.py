from git import Repo
import os

def get_repo_structure(repo_path):
    repo = Repo(repo_path)
    tree = repo.tree()
    top_level_files, top_level_dirs = traverse_tree(tree, repo_path, 0)
    return top_level_files, top_level_dirs

def traverse_tree(tree, repo_path, level):
    files = []
    dirs = []
    for entry in tree.trees:
        dir_path = os.path.join(repo_path, entry.name)
        sub_files, sub_dirs = traverse_tree(entry, dir_path, level + 1)
        dir_info = {
            'path': dir_path,
            'size': sum(d['size'] for d in sub_dirs) + sum(f['size'] for f in sub_files),
            'levels': level + 1,
            'files': len(sub_files),
            'extensions': list(set(f['extension'] for f in sub_files))
        }
        dirs.append(dir_info)
        dirs.extend(sub_dirs)
    for blob in tree.blobs:
        file_path = os.path.join(repo_path, blob.name)
        file_info = {
            'path': file_path,
            'size': blob.size,
            'extension': os.path.splitext(blob.name)[1][1:] if os.path.splitext(blob.name)[1] else ''
        }
        files.append(file_info)
    return files, dirs


def get_repo_structure_as_text(local_repo_path):
    top_level_files, top_level_dirs = get_repo_structure(local_repo_path)
    
    out_string = "Top-level files:"
    for file_info in top_level_files:
        file_info_string = f"{file_info['path']} ({file_info['size']} bytes, {file_info['extension']})"
        out_string = out_string + "\n" + file_info_string

    out_string = out_string + "\n\nTop-level directories:"
    for dir_info in top_level_dirs:
        directory_info_string = f"{dir_info['path']} ({dir_info['size']} bytes, {dir_info['levels']} levels, {dir_info['files']} files, extensions: {', '.join(dir_info['extensions'])})"
        out_string = out_string + "\n" + directory_info_string
    return out_string


def load_readme_file(directory):
    readme_path = os.path.join(directory, "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            readme_content = f.read()
        return readme_content
    else:
        return "No README file found in the specified directory."