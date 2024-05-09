
from langchain_community.llms import ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


## prompt template
default_system_prompt = """
You are a useful AI assitant, who is an expert at reviewing code repositories. 
Please keep your answers consise and strive to answer the users questions exactly.
"""

def get_chain(model = 'llama3', system_prompt = default_system_prompt, user_prompt = "{query}"):
    prompt = ChatPromptTemplate.from_messages(
    [
        ('system', system_prompt),
        ("human", user_prompt),
    ]
)
    llm = ollama.Ollama(
        model=model, 
        stop=['<|eot_id|>'], num_ctx = 2048, temperature=0)
    output_parser = StrOutputParser()
    chain = prompt|llm|output_parser
    return chain