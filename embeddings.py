import os
from pinecone import Pinecone
from langchain.vectorstores import Pinecone as LangchainPinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate,PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.globals import set_debug
from dotenv import load_dotenv
import streamlit as st

set_debug(True)
load_dotenv()


def query_llm(query):
    template = """You're a friendly, empathetic, and knowledgeable customer relationship manager assisting HSBC customers with their queries about migrating their accounts to RBC Bank. Your responses should be based on the information provided in the context, ensuring accuracy and relevance. When answering questions, provide clear and helpful responses that align with the RBC product Migration guide, without explicitly referencing the knowledge base. Instead, weave in the information naturally to enhance customer understanding. 

{context}
Below is the history of conversation before this question :{chat_history}

Question: {question}

Detailed Answer:"""
    custom_rag_prompt = PromptTemplate.from_template(template)
    load_dotenv()
    index_name = 'rbchsbc-retrieval-augmentation'
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_KEY'))
    memory = ConversationBufferMemory(memory_key="chat_history")
    llm = ChatOpenAI(model_name="gpt-4o-mini",memory=memory , prompt= custom_rag_prompt, temperature=0, openai_api_key=os.getenv("OPENAI_KEY"))

    PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]   # os.getenv("PINECONE_API_KEY")

    # Initialize Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    # Initialize the vector store
    docsearch = LangchainPinecone.from_existing_index(index_name, embeddings)
    # docsearch = Pinecone.from_existing_index(index_name, embeddings)
    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
        #     | custom_rag_prompt
            | llm
            | StrOutputParser()
    )
    response = rag_chain.invoke(query)
    sources = retriever.get_relevant_documents(query)
    return response, sources
