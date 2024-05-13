from dotenv import load_dotenv
from langchain_community.llms import ollama, HuggingFaceEndpoint
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser

import os

load_dotenv()

## prompt template
default_system_prompt = """
You are a useful AI assitant, who is an expert at reviewing code repositories. 
Please keep your answers consise and strive to answer the users questions exactly.
"""

def get_chain(framework='ollama', model = 'llama3', prompt ='{query}'):
    if framework == 'ollama':
        llm = get_ollama_model(model=model)
    elif framework == 'openai':
        llm = get_openai_model(model=model)
    elif framework == 'huggingface':
        llm = get_hugginface_model(model=model)
    else:
        raise NotImplementedError
    
    prompt_template = get_prompt_template(prompt=prompt)
    return prompt_template | llm | StrOutputParser()

def get_ollama_model(model = 'llama3'):
    llm = ollama.Ollama(
        model=model, 
        stop=['<|eot_id|>'], 
        num_ctx = 2048, 
        temperature=0)
    return llm

def get_openai_model(model='gpt-3.5-turbo'):
    llm = ChatOpenAI(
        model=model, 
        temperature=0)
    return llm

def get_hugginface_model(model="mistralai/Mistral-7B-Instruct-v0.2"):
    llm = HuggingFaceEndpoint(
        repo_id=model,  temperature=0.01, model_kwargs=dict(stop_token=['<|eot_id|>'], max_length=1024, token=os.environ.get('HUGGINGFACEHUB_API_TOKEN')))
    return llm

def get_prompt_template(prompt):
    prompt_template = PromptTemplate.from_template(prompt)

    return prompt_template

