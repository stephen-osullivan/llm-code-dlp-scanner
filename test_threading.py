import concurrent.futures
import time
from chains import get_chain

chain = get_chain(framework='openai')

def get_response(query: str) -> str:
    return chain.invoke({'query': query})

queries = [
    'what is love?', 'who was the most powerful queen of england?', 'How can I learn how to use asyncio in python?',
    'Tell me the history of Islam.', 'Where are the wild things?', 'is Asynio faster than threading for rest api calls?'
]
t0 = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    # Submit the tasks to the executor
    threads = []
    for i, q in enumerate(queries):
        threads.append(executor.submit(get_response, q))
        print('Submited thread', i)

    # print results as and when they come in
    for future in concurrent.futures.as_completed(threads):
        print(future.result())

print('Total Execution time:', time.time()-t0)