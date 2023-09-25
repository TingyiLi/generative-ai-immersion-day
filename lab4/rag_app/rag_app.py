import json
import os
from langchain.chains import ConversationalRetrievalChain
from langchain import SagemakerEndpoint
from langchain.prompts.prompt import PromptTemplate
from langchain.embeddings import SagemakerEndpointEmbeddings
from langchain.embeddings.sagemaker_endpoint import EmbeddingsContentHandler
from langchain.llms.sagemaker_endpoint import ContentHandlerBase, LLMContentHandler
from langchain.memory import ConversationBufferWindowMemory
from langchain import PromptTemplate, LLMChain
from langchain.memory.chat_message_histories import DynamoDBChatMessageHistory
# from kendra.kendra_index_retriever import KendraIndexRetriever
from langchain.retrievers import AmazonKendraRetriever
from langchain.chains.question_answering import load_qa_chain
from typing import Dict


REGION = os.environ.get('REGION')
KENDRA_INDEX_ID = os.environ.get('KENDRA_INDEX_ID')
SM_ENDPOINT_NAME = os.environ.get('SM_ENDPOINT_NAME')

# Generative LLM 

# Content Handler for Option 1 - FLAN-T5-XXL - please uncomment below if you used this option
# class ContentHandler(LLMContentHandler):
#     content_type = "application/json"
#     accepts = "application/json"

#     def transform_input(self, prompt, model_kwargs):
#         input_str = json.dumps({"text_inputs": prompt, "temperature": 0, "max_length": 200})
#         return input_str.encode('utf-8')
    
#     def transform_output(self, output):
#         response_json = json.loads(output.read().decode("utf-8"))
#         return response_json["generated_texts"][0]

# Content Handler for Option 2 - Falcon40b-instruct - please uncomment below if you used this option
# class ContentHandler(LLMContentHandler):
#     content_type = "application/json"
#     accepts = "application/json"

#     def transform_input(self, prompt, model_kwargs):
#         input_str = json.dumps({
#             "inputs": prompt, 
#             "parameters": {
#                 "do_sample": False, 
#                 "repetition_penalty": 1.1, 
#                 "return_full_text": False, 
#                 "max_new_tokens":100
#             }
#         })
#         return input_str.encode('utf-8')
    
#     def transform_output(self, output):
#         response_json = json.loads(output.read().decode("utf-8"))
#         return response_json[0]["generated_text"]
    
# Content Handler for Option3 - Falcon 7B
# class ContentHandler(LLMContentHandler):
#     content_type = "application/json"
#     accepts = "application/json"

#     def transform_input(self, prompt, model_kwargs):
#         input_str = json.dumps({
#             "inputs": prompt, 
#             "parameters": {
#                 "do_sample": False, 
#                 "top_p": 0.9,
#                 "temperature": 0.8,
#                 "max_new_tokens": 100,
#                 "stop": ["<|endoftext|>", "</s>"]
#             }
#         })
#         return input_str.encode('utf-8')
    
#     def transform_output(self, output):
#         print("output:",output)
#         response_json = json.loads(output.read().decode("utf-8"))
#         return response_json[0]["generated_text"]

# content_handler = ContentHandler()

# # SageMaker langchain integration, to assist invoking SageMaker endpoint.
# llm=SagemakerEndpoint(
#     endpoint_name=SM_ENDPOINT_NAME,
# #    model_kwargs=kwargs,
#     region_name=REGION,
#     content_handler=content_handler, 
# )

# _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language. 

# Chat History:
# {chat_history}
# Follow Up Input: {question}
# Standalone question:"""
# CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)


def lambda_handler(event, context):
    print(event)
    body = json.loads(event['body'])
    print(body)
    query = body['query']
    uuid = body['uuid']
    print(query)
    print(uuid)

    message_history = DynamoDBChatMessageHistory(table_name="MemoryTable", session_id=uuid)
    memory = ConversationBufferWindowMemory(memory_key="chat_history", chat_memory=message_history, return_messages=True, k=3)
    
    # This retriever is using the query API, self implement
    # retriever = KendraIndexRetriever(kendraindex=KENDRA_INDEX_ID, 
    #                                  awsregion=REGION, 
    #                                  return_source_documents=True)
    
    # This retriever is using the new Kendra retrieve API https://aws.amazon.com/blogs/machine-learning/quickly-build-high-accuracy-generative-ai-applications-on-enterprise-data-using-amazon-kendra-langchain-and-large-language-models/
    retriever = AmazonKendraRetriever(
        index_id=KENDRA_INDEX_ID,
        region_name=REGION,
        top_k = 2
    )
    
    retriever.get_relevant_documents(query)
    
    qa = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory, condense_question_prompt=CONDENSE_QUESTION_PROMPT, verbose=True)

    response = qa.run(query)
    clean_response = response['output_text'].replace('\n','').strip()
    
    prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    Question: {question}
    Helpful Answer:"""
    QA_PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )


    class ContentHandler(LLMContentHandler):
        content_type = "application/json"
        accepts = "application/json"

        def transform_input(self, prompt: str, model_kwargs: Dict) -> bytes:
            input_str = json.dumps({"inputs": prompt, **model_kwargs})
            print(json.dumps(json.loads(input_str), indent=2))
            return input_str.encode("utf-8")

        def transform_output(self, output: bytes) -> str:
            response_json = json.loads(output.read().decode("utf-8"))
            return response_json[0]['generated_text']


    content_handler = ContentHandler()

    chain = load_qa_chain(
        llm=SagemakerEndpoint(
            endpoint_name=SM_ENDPOINT_NAME,
            region_name=REGION,
            model_kwargs={
                "parameters": {
                    "do_sample": True,
                    "top_p": 0.9,
                    "temperature": 0.001, 
                    "max_length": 512,
                    "stop": ["<|endoftext|>", "</s>"]
                }
            },
            content_handler=content_handler,
        ),
        prompt=QA_PROMPT,
    )
    
    response = chain({"input_documents": docs , "question": query}, return_only_outputs=True)
    clean_response = response['output_text'].replace('\n','').strip().split("Helpful Answer: ")[1]

    return {
        'statusCode': 200,
        'body': json.dumps(f'{clean_response}')
        }
