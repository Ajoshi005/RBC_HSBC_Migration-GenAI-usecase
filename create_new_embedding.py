from pinecone import Pinecone as PineconeClient, PodSpec
from langchain_community.vectorstores import Pinecone
from langchain_community.document_loaders import PyPDFDirectoryLoader
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.globals import set_debug
from dotenv import load_dotenv

set_debug(True)
load_dotenv()

# Load pdf files using directory loader
# Txt files store under directory
dir_path = "/Users/akashjoshi/Desktop/Python_Learning/RBC_HSBC_Migration-GenAI-usecase/Docs_to_be_loaded"

# Load, chunk and index the contents of the blog.
loader = PyPDFDirectoryLoader(dir_path)
docs = loader.load()

# 219 docs
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
splits = text_splitter.split_documents(docs)
# 802 chunks of 400 tokens


embed_model = OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_KEY'))  # os.getenv('OPENAI_KEY'))


# vectorstore = Chroma.from_documents(documents=splits, embedding=embed_model)
# --------------PINECONE DB Implementation------------------

# Already Run below statement once
def create_knowledge_base(split_docs):
    # Function creates an index in pinecone(if it doesnt exist) and creates a vectorstore of the knowledge base.
    # gets list of docs(chunks) as input and Returns retreiver

    index_name = 'rbchsbc-retrieval-augmentation'
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

    # init
    pinecone = PineconeClient(api_key=PINECONE_API_KEY, environment='us-east-1')

    # if index_name not in pinecone.list_indexes():
    #     pinecone.create_index(
    #         name=index_name,
    #         metric='cosine',
    #         spec=PodSpec(environment='us-west1-gcp-free'),
    #         dimension=1536  # 1536 dim of text-embedding-ada-002
    #     )

    # embedding model from OpenAI
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_KEY'))

    vectordb = Pinecone.from_documents(split_docs, embeddings, index_name=index_name)
    retriever = vectordb.as_retriever()
    return retriever


# delete the pinecone index content
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pc = PineconeClient(api_key=PINECONE_API_KEY, environment='us-east-1')
index = pc.Index("rbchsbc-retrieval-augmentation")
index.delete(delete_all=True)


# Call the create knowledge function(High OpenAI-Usage call)
retriever_check = create_knowledge_base(splits)

# check retriever
print(retriever_check.get_relevant_documents("What changes to expect on my credit card fees?"))
