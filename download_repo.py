import os

from github import Github

def download_github_repo(repo_name, dir = None):
    g=Github(login_or_token=os.environ.get('GITHUB_TOKEN'))
    repo = g.get_repo(repo_name)
    repo_contents = repo.get_contents("")
    if dir:
        os.makedirs(dir, exist_ok=True)
    for file in repo_contents:
        try: 
            if file.type == "file":
                file_name = file.name
                file_content = repo.get_contents(file_name).decoded_content.decode("utf-8")
                with open(os.path.join(dir, file_name), "w") as f:
                    f.write(file_content)
        except Exception as e:
            print(f'Failed to download {file}. Got exception:', e)


def load_github_dir_into_text_file(github_dir, output_file):
    """
    Load each file from a local GitHub directory into a single text file.

    Args:
        github_dir (str): Path to the local GitHub directory
        output_file (str): Path to the output text file

    Returns:
        None
    """
    with open(output_file, "w", encoding="utf-8") as output:
        # Start with the README file
        readme_file = os.path.join(github_dir, "README.md")
        if os.path.exists(readme_file):
            with open(readme_file, "r", encoding="utf-8") as f:
                output.write(f.read() + "\n")

        # Iterate over the files in the directory
        for root, dirs, files in os.walk(github_dir):
            if '.git' in root.split('/'):
                continue
            for file in files:
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    output.write("#" * 80 + "\n")
                    output.write(f"# {file_path}\n")
                    output.write("#" * 80 + "\n")
                    if file_path!= readme_file and file_path.split('.')[-1] != 'ipynb':  # Skip the README file
                        try: 
                            output.write(f.read() + "\n\n")
                        except Exception as e:
                            print(f'Not able to load file: {os.path.join(dirs,file)}. Got Exception: {e}')
                            output.write(f"Not able to load file: {file_path}. Got Exception: {e}" "\n\n")
                        

    print(f"Files loaded into {output_file}")


def load_github_repo(repo_owner, repo_name, github_token=None):
    """
    Load a GitHub repository and prepare it for LLM ingestion.

    Args:
        repo_owner (str): GitHub repository owner (e.g., 'username' or 'organization')
        repo_name (str): GitHub repository name (e.g.,'my-repo')
        github_token (str): GitHub personal access token with repo access

    Returns:
        repo_data (dict): Dictionary containing the repository data, including:
            - repo_name (str): Repository name
            - repo_description (str): Repository description
            - files (list): List of file paths and contents
            - commits (list): List of commit messages and authors
    """
    # Initialize the GitHub API client
    g = Github(github_token)

    # Get the repository object
    repo = g.get_repo(f"{repo_owner}/{repo_name}")

    # Initialize the repository data dictionary
    repo_data = {
        "repo_name": repo_name,
        "repo_description": repo.description,
        "files": [],
        "commits": []
    }

    # Iterate over the repository files
    for file in repo.get_contents(""):
        # Get the file contents and path
        file_contents = file.decoded_content.decode("utf-8")
        file_path = file.path

        # Add the file data to the repository data dictionary
        repo_data["files"].append({"path": file_path, "contents": file_contents})

    # Iterate over the repository commits
    for commit in repo.get_commits():
        # Get the commit message and author
        commit_message = commit.commit.message
        commit_author = commit.commit.author.name

        # Add the commit data to the repository data dictionary
        repo_data["commits"].append({"message": commit_message, "author": commit_author})

    return repo_data


