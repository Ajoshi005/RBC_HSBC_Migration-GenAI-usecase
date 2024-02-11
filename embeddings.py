from pinecone import Pinecone as PineconeClient, PodSpec
from langchain_community.vectorstores import Pinecone
from langchain_community.document_loaders import PyPDFDirectoryLoader
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.globals import set_debug
from dotenv import load_dotenv
import streamlit as st

set_debug(True)
load_dotenv()

# Load pdf files using directory loader
# Txt files store under directory
dir_path = "/Users/akashjoshi/Desktop/Python_Learning/RBC_HSBC_Migration GenAI usecase/Docs_to_be_loaded"

# Load, chunk and index the contents of the blog.
loader = PyPDFDirectoryLoader(dir_path)
docs = loader.load()

# 219 docs
text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
splits = text_splitter.split_documents(docs)
# 2015 chunks of 400 tokens


embed_model = OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_KEY'))  # os.getenv('OPENAI_KEY'))


# vectorstore = Chroma.from_documents(documents=splits, embedding=embed_model)
# --------------PINECONE DB Implementation------------------

# Already Run below statement once
def create_knowledge_base(split_docs):
    # Function creates an index in pinecone(if it doesnt exist) and creates a vectorstore of the knowledge base.
    # gets list of docs(chunks) as input and Returns retreiver

    index_name = 'rbchsbc-retrieval-augmentation'
    PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]  # os.getenv("PINECONE_API_KEY")

    # init
    pinecone = PineconeClient(api_key=PINECONE_API_KEY, environment='us-west1-gcp-free')

    if index_name not in pinecone.list_indexes():
        pinecone.create_index(
            name=index_name,
            metric='cosine',
            spec=PodSpec(environment='us-west1-gcp-free'),
            dimension=1536  # 1536 dim of text-embedding-ada-002
        )

    # embedding model from OpenAI
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_KEY'))

    vectordb = Pinecone.from_documents(split_docs, embeddings, index_name=index_name)
    retriever = vectordb.as_retriever()
    return retriever


def query_llm(query):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=os.getenv('OPENAI_KEY'))

    template = """You're a helpful, empathetic and friendly customer relationship manager helping respond to HSBC customer 
    queries regarding the migration of their accounts to RBC Bank.
    You rely on the knowledge from the context for your replies.When answering questions, be sure to provide answers that 
    reflect the content of the knowledge base,but avoid saying things like 'according to the knowledge base'. 
    Instead, subtly mention that the information is based on the RBC product Migration guide.
    {context}
    Question: {question}
    Detailed Helpful Answer:"""
    custom_rag_prompt = ChatPromptTemplate.from_template(template)

    index_name = 'rbchsbc-retrieval-augmentation'
    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_KEY"])

    PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]  # os.getenv("PINECONE_API_KEY")

    # init
    pinecone = PineconeClient(api_key=PINECONE_API_KEY, environment='us-west1-gcp-free')

    docsearch = Pinecone.from_existing_index(index_name, embeddings)
    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | custom_rag_prompt
            | llm
            | StrOutputParser()
    )
    response = rag_chain.invoke(query)
    sources = retriever.get_relevant_documents(query)
    return response, sources
