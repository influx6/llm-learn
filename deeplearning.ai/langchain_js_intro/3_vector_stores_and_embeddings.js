# Lesson 3: Vectorstores and embeddings

# Vectorstore ingestion

import "dotenv/config";

import { OpenAIEmbeddings } from "@langchain/openai";

const embeddings = new OpenAIEmbeddings();

await embeddings.embedQuery("This is some sample text");

import { similarity } from "ml-distance";

const vector1 = await embeddings.embedQuery(
    "What are vectors useful for in machine learning?"
);
const unrelatedVector = await embeddings.embedQuery(
    "A group of parrots is called a pandemonium."
);

similarity.cosine(vector1, unrelatedVector);

const similarVector = await embeddings.embedQuery(
    "Vectors are representations of information."
);

similarity.cosine(vector1, similarVector);

// Peer dependency
import * as parse from "pdf-parse";
import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import { 
    RecursiveCharacterTextSplitter
} from "langchain/text_splitter";

const loader = new PDFLoader("./data/MachineLearning-Lecture01.pdf");

const rawCS229Docs = await loader.load();

const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 128,
  chunkOverlap: 0,
});

const splitDocs = await splitter.splitDocuments(rawCS229Docs);

import { MemoryVectorStore } from "langchain/vectorstores/memory";

const vectorstore = new MemoryVectorStore(embeddings);

await vectorstore.addDocuments(splitDocs);

const retrievedDocs = await vectorstore.similaritySearch(
    "What is deep learning?", 
    4
);

const pageContents = retrievedDocs.map(doc => doc.pageContent);

pageContents

# Retrievers

const retriever = vectorstore.asRetriever();

await retriever.invoke("What is deep learning?")























