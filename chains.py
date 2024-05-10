from dotenv import load_dotenv
from langchain_community.llms import ollama
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

## prompt template
default_system_prompt = """
You are a useful AI assitant, who is an expert at reviewing code repositories. 
Please keep your answers consise and strive to answer the users questions exactly.
"""

def get_chain(framework='ollama', model = 'llama3', prompt = "{query}"):
    if framework == 'ollama':
        return get_ollama_chain(model=model, prompt=prompt)
    elif framework == 'openai':
        return get_openai_chain(model=model, prompt=prompt)
    else:
        raise NotImplementedError

def get_ollama_chain(model = 'llama3', prompt = "{query}"):
    prompt = PromptTemplate.from_template(prompt)
    llm = ollama.Ollama(
        model=model, 
        stop=['<|eot_id|>'], 
        num_ctx = 2048, 
        temperature=0)
    output_parser = StrOutputParser()
    chain = prompt|llm|output_parser
    return chain

def get_openai_chain(model='gpt-3.5-turbo', prompt = "{query}"):
    prompt = PromptTemplate.from_template(prompt)
    llm = ChatOpenAI(
        model=model, 
        temperature=0)
    output_parser = StrOutputParser()
    chain = prompt|llm|output_parser
    return chain