
import os
import openai
import sys
sys.path.append('../..')

from langchain.document_loaders import PyPDFLoader
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.environ['OPENAI_API_KEY']

from langchain.document_loaders import PyPDFLoader

# Load PDF
loaders = [
    # Duplicate documents on purpose - messy data
    PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture01.pdf"),
    PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture01.pdf"),
    PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture02.pdf"),
    PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture03.pdf")
]
docs = []
for loader in loaders:
    docs.extend(loader.load())


# Split Documents
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 150
)

splits = text_splitter.split_documents(docs)

len(splits)

# Embeddings

from langchain.embeddings.openai import OpenAIEmbeddings
embedding = OpenAIEmbeddings()

sentence1 = "i like dogs"
sentence2 = "i like canines"
sentence3 = "the weather is ugly outside"

embedding1 = embedding.embed_query(sentence1)
embedding2 = embedding.embed_query(sentence2)
embedding3 = embedding.embed_query(sentence3)

import numpy as np

np.dot(embedding1, embedding2)
np.dot(embedding1, embedding3)
np.dot(embedding2, embedding3)

# VectorStores
from langchain.vectorstores import Chroma

persist_directory = 'docs/chroma/'

# rm -rf ./docs/chroma  # remove old database files if any

vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory=persist_directory
)

print(vectordb._collection.count())

# Similarity Search

question = "is there an email i can ask for help"

docs = vectordb.similarity_search(question,k=3)

len(docs)

docs[0].page_content

vectordb.persist()

### Failure modes

""" 
This seems great, and basic similarity search will get you 80% of the way there very easily.

But there are some failure modes that can creep up.

Here are some edge cases that can arise - we'll fix them in the next class.
"""

question = "what did they say about matlab?"

docs = vectordb.similarity_search(question,k=5)

"""
Notice that we're getting duplicate chunks (because of the duplicate MachineLearning-Lecture01.pdf in the index).

Semantic search fetches all similar documents, but does not enforce diversity.

docs[0] and docs[1] are indentical.
"""

docs[0]
docs[1]

"""
We can see a new failure mode.

The question below asks a question about the third lecture, but includes results from other lectures as well.
"""

question = "what did they say about regression in the third lecture?"
docs = vectordb.similarity_search(question,k=5)

for doc in docs:
    print(doc.metadata)

print(docs[4].page_content)