import gradio as gr
import os, pinecone
from datetime import datetime

from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.llms import OpenAI
from jinja2 import Template


OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENV = os.getenv('PINECONE_ENV')


template = """You are an AI assistant for Mahabharat. Your name is Kala, which is Sanskrit for Time. You are given the following extracted parts of a long document and a question. If the question is not about Mahabharat, please do not answer stating you are a AI Assistant focused on Mahabharat.

QUESTION: {question}
=========
{summaries}
=========
FINAL ANSWER IN English
"""

index_name = "ntg-demo"
namespace = "mahabharat"
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

PROMPT = PromptTemplate(template=template, input_variables=["summaries", "question"])
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
docsearch = Pinecone.from_existing_index(index_name, embeddings, namespace=namespace)

def search_datastore(query):
    chat_history=[]
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Timestamp: {timestamp} : {query}")
    chain = load_qa_with_sources_chain(OpenAI(temperature=0, model_name="text-davinci-003", n=3, best_of=3), chain_type="stuff", prompt=PROMPT)
    docs_test = docsearch.similarity_search(query, include_metadata=True, k=3,)
    results = chain({"input_documents": docs_test, "question": query})    
    datareturn = results["output_text"]
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Timestamp: {timestamp} : {datareturn}")
    chat_history.append(f"[User]:\n{query}\n")
    chat_history.append(f"[MahabharatAI]\n{datareturn}\n")
    return "\n".join(chat_history)

def render_chat_history(chat_history):
    rendered_chat = []
    for sender, message in chat_history:
        rendered_chat.append(f"{sender}: {message}")
    return "\n".join(rendered_chat)

iface = gr.Interface(
    fn=search_datastore,
    inputs=gr.inputs.Textbox(lines=2, placeholder="Enter your query here...", label="User Message"),
    outputs=gr.outputs.Textbox(label="Chat History"),
    title="Mahabharat Chatbot",
    description="A chatbot interface to query the Mahabharat text from a local datastore.",
    allow_multiple_users=True,
    theme="huggingface",
)

iface.launch()