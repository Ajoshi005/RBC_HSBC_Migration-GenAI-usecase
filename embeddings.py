from pinecone import Pinecone as PineconeClient, PodSpec
from langchain_community.vectorstores import Pinecone
import os
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.globals import set_debug
from dotenv import load_dotenv
import streamlit as st

set_debug(True)
load_dotenv()


def query_llm(query):
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, openai_api_key=os.getenv("OPENAI_KEY"))

    template = """You're a helpful, empathetic and friendly customer relationship manager helping respond to HSBC customer 
    queries regarding the migration of their accounts to RBC Bank.
    You rely on the knowledge from the context for your replies.When answering questions, be sure to provide answers that 
    reflect the content of the knowledge base,but avoid saying things like 'according to the knowledge base'. 
    Instead, subtly mention that the information is based on the RBC product Migration guide.
    {context}
    Question: {question}
    Detailed Answer:"""
    custom_rag_prompt = ChatPromptTemplate.from_template(template)
    load_dotenv()
    index_name = 'rbchsbc-retrieval-augmentation'
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_KEY'))

    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")  # st.secrets["PINECONE_API_KEY"]

    # init
    pinecone = PineconeClient(api_key=PINECONE_API_KEY, environment='us-east-1')
    Pinecone.init(
        api_key=os.getenv('PINECONE_API_KEY'),
        environment="YOUR_PINECONE_ENVIRONMENT"
    )

    docsearch = Pinecone.from_existing_index(index_name, embeddings)
    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | custom_rag_prompt
            | llm
            | StrOutputParser()
    )
    response = rag_chain.invoke(query)
    sources = retriever.get_relevant_documents(query)
    return response, sources
