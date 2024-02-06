import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.globals import set_debug
from dotenv import load_dotenv

set_debug(True)

# Load the env vars
load_dotenv()

ChatOpenAI.api_key = os.getenv('OPENAI_KEY')

# Load pdf files using directory loader
# Txt files store under directory
dir_path = "/Users/akashjoshi/Desktop/Python_Learning/RBC_HSBC_Migration GenAI usecase/Docs_to_be_loaded"

# Load, chunk and index the contents of the blog.
loader = PyPDFDirectoryLoader(dir_path)
docs = loader.load()

# 219 docs
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
print(len(splits))  # 848 chunks of 1000 tokens

embed_model = OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_KEY'))
vectorstore = Chroma.from_documents(documents=splits, embedding=embed_model)

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=os.getenv('OPENAI_KEY'))

template = """You're a helpful, empathetic and friendly customer relationship manager at RBC Bank 
and rely on the knowledge from the context.When answering questions, be sure to provide answers that 
reflect the content of the knowledge base,but avoid saying things like 'according to the knowledge base'. 
Instead, subtly mention that the information is based on the RBC product guide.
{context}
Question: {question}
Helpful Answer:"""
custom_rag_prompt = ChatPromptTemplate.from_template(template)

rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
        | StrOutputParser()
)


def generate_response(question):
    # Assuming rag_chain is defined in your function.py
    response = rag_chain.invoke(question)
    return response


# ----------BUILDING the APP.py----------------------------#

# Set Streamlit app title with emojis of a parrot and a chain
st.title("RBC Transition  AI Assistant ")

# Define the colors for the RBC Bank Canada color palette
rbc_canada_palette = {
    "primary": "#003366",  # RBC Blue
    "secondary": "#FF7F00",  # RBC Orange
    "background": "#FFFFFF",  # White
    "text": "#333333",  # Dark Gray
}
# Set page background color and text color using RBC Bank Canada color palette
st.markdown(
    f"""
    <style>
    body {{
        background-color: {rbc_canada_palette["background"]};
        color: {rbc_canada_palette["text"]};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Get user input from text area
user_input = st.text_area("Please enter your query on transition of your HSBC accounts to RBC:")

# Check if the user has entered any input
if user_input:
    # Process user input and get the model's output
    model_output = generate_response(user_input)

    # Display the output in formatted text format
    st.markdown(
        f"""
        <div style='background-color: {rbc_canada_palette["secondary"]}; padding: 10px;'>
            <p style='color: {rbc_canada_palette["primary"]}; font-size: 18px;'>
                {model_output}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
# Add a heading to the sidebar
st.sidebar.header("About the App")

# Add a description to the sidebar
st.sidebar.write(
    "This Langchain based App breaks the RBC Product guide(public) into chunks of text and stores on a Chroma vector DB(created with OpenAi embedding model)."
    "It accepts user queries and searches for relevant subsections from document to pass into OpenAI LLM QA model to get a response"
)
sidebar_text = """
Tech stack:
1) LangChain - for LLM App implementation
2) Open AI - For vector embedding and LLM for creating responses
3) ChromaDB - Creating and storing text data in vector embedding for similarity search(cosine)
4) Streamlit - For App UI and hosting
"""

st.sidebar.markdown(sidebar_text)

# Add a link to your resume on LinkedIn in the sidebar
linkedin_url = "https://www.linkedin.com/in/akashjoshi/"
st.sidebar.markdown(
    '<a href="' + linkedin_url + '"><img src="https://content.linkedin.com/content/dam/me/business/en-us/amp/brand-site/v2/bg/LI-Bug.svg.original.svg" width=30 height=30>',
    unsafe_allow_html=True)
# GitHub Logo and Link
github_url = "https://github.com/Ajoshi005"
st.sidebar.markdown(
    '<a href="' + github_url + '"><img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width=30 height=30></a>',
    unsafe_allow_html=True)

# Add the disclaimer at the bottom of the page
st.markdown(
    """
    #### Disclaimer:
    The information provided by this application is for educational purposes only and should not be considered
     as legal advice. All answers to the queries are generated by an AI model and may not be entirely accurate or 
     up-to-date. Please speak to a relationship manager for any specific details relating to your account before making any changes.
    """,
    unsafe_allow_html=True,
)
