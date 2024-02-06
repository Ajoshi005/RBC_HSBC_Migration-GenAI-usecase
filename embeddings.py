from pinecone import Pinecone as PineconeClient
from langchain_community.document_loaders import PyPDFDirectoryLoader
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.globals import set_debug


set_debug(True)

# Load the env vars
from dotenv import load_dotenv
load_dotenv()

# Load pdf files using directory loader
# Txt files store under directory
dir_path = "/Users/akashjoshi/Desktop/Python_Learning/RBC_HSBC_Migration GenAI usecase/Docs_to_be_loaded"

# Load, chunk and index the contents of the blog.
loader = PyPDFDirectoryLoader(dir_path)
docs = loader.load()

# 219 docs
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
print(len(splits))
#848 chunks of 1000 tokens

embed_model = OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_KEY'))
vectorstore = Chroma.from_documents(documents=splits, embedding=embed_model)

#--------------PINECONE DB Implementation------------------

#pip install "pinecone-client[grpc]"
index_name = 'RBCHSBC-retrieval-augmentation'
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

#init
pinecone = PineconeClient(api_key=PINECONE_API_KEY, environment='us-west1-gcp-free')

if index_name not in pinecone.list_indexes():
    # we create a new index
    pinecone.create_index(
        name=index_name,
        metric='cosine',
        dimension=1536  # 1536 dim of text-embedding-ada-002
    )

#embedding model from OpenAI
embed_model = OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_KEY'))

#Already Run below statement once
#crasearch = Pinecone.from_texts(texts=splits, embedding=embed_model, index_name=index_name)
text_field = "text"

index1 = pinecone.Index(index_name)

vectorstore = Pinecone(
    index1, embed.embed_query, text_field
)