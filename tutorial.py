## Pre-deployment setup
## Create a virtual environment using Python 3.10
# python3.10 -m venv .venv
## Activate the virtual environment
# source .venv/bin/activate
## Upgrade pip
# pip install --upgrade pip
## Install required packages
# pip install langchain pinecone-client openai tiktoken unstructured pdf2image pytesseract

# Edit the sample.env file and use the following command to set the variables
# source sample.env

# Import os to access environment variables
import os

# Get the API keys and Pinecone environment from the environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENV = os.getenv('PINECONE_ENV')

# Download and extract the Mahabharat text
# Download link: https://library.bjp.org/jspui/handle/123456789/851
# Extract the contents into a folder called 'mahabharat' in the same location as this file

# Import DirectoryLoader to load the text files from the 'mahabharat' folder
from langchain.document_loaders import DirectoryLoader

# Create a DirectoryLoader instance and load the documents
loader = DirectoryLoader('mahabharat/', glob="**/*.txt")
docs = loader.load()

# Check how many files were loaded
len(docs)

# Split the documents into chunks for storage in a vector database
# We'll use Pinecone as our datastore, but Langchains Chroma can also be used
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Create a text splitter instance
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, length_function=len)

# Split the documents
document = text_splitter.split_documents(docs)

# Use OpenAI Embeddings for text representation
# Note: Using OpenAI API may incur costs, so ensure you have access to the API
from langchain.embeddings.openai import OpenAIEmbeddings

# Create an OpenAIEmbeddings instance
embeddings = OpenAIEmbeddings()

# Alternative: Use HuggingFace Embeddings
# from langchain.embeddings.openai import HuggingFaceHubEmbeddings
# embeddings = HuggingFaceHubEmbeddings()

# Set up Pinecone vector store
# Register for a free index on http://pinecone.io
# Metric: cosine
# Dimensions: 1536
from langchain.vectorstores import Pinecone
import pinecone

# Set the index and namespace for Pinecone
index_name = "ntg-demo"
namespace = "mahabharat-demo"

# Initialize Pinecone instance with the API key and environment
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

# Load the Mahabharat data and embeddings into Pinecone vector store - This takes a while - around 10-20 minutes depending on your upload speed.
docsearch = Pinecone.from_documents(document, embeddings, index_name=index_name, namespace=namespace)

# Alternative: If data is already loaded, use the existing index
# docsearch = Pinecone.from_existing_index(index_name, embeddings, namespace=namespace)

# Use OpenAI's GPT-3 (text-davinci-003) as the language model
from langchain.llms import OpenAI

# Create an OpenAI instance with specified parameters
llm = OpenAI(temperature=0.5, model_name="text-davinci-003", n=2, best_of=2)

# Load the question-answering chain
from langchain.chains.question_answering import load_qa_chain
chain = load_qa_chain(llm, chain_type="stuff")

# Create a query and search the Pinecone vector store for relevant results
query = "Whos fault was it the war started?"
docs_test = docsearch.similarity_search(query, include_metadata=True)
chain.run(input_documents=docs_test, question=query)

# Create another query and search the Pinecone vector store
query = "Why was it Dhritarashtra's fault"
docs_test = docsearch.similarity_search(query, include_metadata=True)
chain.run(input_documents=docs_test, question=query)

# Create a query to generate a poem about the Mahabharat
query = "Create a poem about the mahabharat"
docs_test = docsearch.similarity_search(query, include_metadata=True)
chain.run(input_documents=docs_test, question=query)

# Create a query to generate a short story for children from the Mahabharat
query = "Create a short story for children from the mahabharat"
docs_test = docsearch.similarity_search(query, include_metadata=True)
r = chain.run(input_documents=docs_test, question=query)
print(r)

# Build on the above by using prompt engineering for smarter answers
# Change the language of the final answer if desired
from langchain.prompts import PromptTemplate
template = """You are an AI assistant for Mahabharat. Your name is Kala, which is Sanskrit for Time. You are given the following extracted parts of a long document and a question. If the question is not about Mahabharat, please do not answer stating you are a AI Assistant focused on Mahabharat.

QUESTION: {question}
=========
{summaries}
=========
FINAL ANSWER IN English
"""

# Create the prompt
PROMPT = PromptTemplate(template=template, input_variables=["summaries", "question"])

# Create a new chain with the prompt and ask a question
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
chain = load_qa_with_sources_chain(OpenAI(temperature=0), chain_type="stuff", prompt=PROMPT)
query = "When was the Mahabharat written"
docs_test = docsearch.similarity_search(query, include_metadata=True)
chain({"input_documents": docs_test, "question": query}, return_only_outputs=True)


# We will use Gradio to create a user interface (UI) for our application
import gradio as gr
import os, pinecone

# Import necessary components from the langchain library
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.llms import OpenAI

# Define a custom prompt template for the AI assistant
template = """You are an AI assistant for Mahabharat. Your name is Kala, which is Sanskrit for Time. You are given the following extracted parts of a long document and a question. If the question is not about Mahabharat, please do not answer stating you are a AI Assistant focused on Mahabharat.

QUESTION: {question}
=========
{summaries}
=========
FINAL ANSWER IN English
"""

# Set up Pinecone index and namespace
index_name = "ntg-demo"
namespace = "mahabharat"

# Initialize Pinecone with the API key and environment
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

# Create a prompt template with the specified input variables
PROMPT = PromptTemplate(template=template, input_variables=["summaries", "question"])

# Initialize embeddings and Pinecone vector store
embeddings = OpenAIEmbeddings()
docsearch = Pinecone.from_existing_index(index_name, embeddings, namespace=namespace)

# Define a function to search the Pinecone datastore
def search_datastore(query):
    # Create a chain using the prompt template and OpenAI model
    chain = load_qa_with_sources_chain(OpenAI(temperature=0, model_name="text-davinci-003", n=3, best_of=3), chain_type="stuff", prompt=PROMPT)
    
    # Search for relevant documents
    docs_test = docsearch.similarity_search(query, include_metadata=True, k=3,)
    
    # Run the chain and store the results
    results = chain({"input_documents": docs_test, "question": query})
    
    # Format the results for display
    data = "##Answer\n{}\n\n##Source Document\n{}".format(results["output_text"], results["input_documents"])
    
    return data

# Create a Gradio interface with the specified input and output elements
iface = gr.Interface(fn=search_datastore, inputs=gr.inputs.Textbox(lines=2, placeholder="Enter your question here..."), outputs="text", title="Mahabharat Pinecone Datastore Search", description="Ask questions to about the Mahabharat Pinecone vector datastore using Langchain.")

# Launch the Gradio interface
iface.launch()

