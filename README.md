# Data Loss Prevention App

Streamlit langchain app to analyse a repo for DLP.

1) We download a repo using the gitpython package.
2) We then analyse the code in this repo using an LLM that can be selected from a variety of frameworks e.g. openai, hugginface, ollama.
3) We then output the results in a streamlit app.

!['Screenshot of the APP']('repo-scan.png')