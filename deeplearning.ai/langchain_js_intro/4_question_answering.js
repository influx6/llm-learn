# Lesson 4: Question answering

![](./images/rag_diagram.png)

import "dotenv/config";

import { loadAndSplitChunks } from "./lib/helpers.ts";

const splitDocs = await loadAndSplitChunks({
    chunkSize: 1536,
    chunkOverlap: 128
});

import { initializeVectorstoreWithDocuments } from "./lib/helpers.ts";

const vectorstore = await initializeVectorstoreWithDocuments({
  documents: splitDocs,
});

const retriever = vectorstore.asRetriever();

# Document retrieval in a chain

import { RunnableSequence } from "@langchain/core/runnables";
import { Document } from "@langchain/core/documents";

const convertDocsToString = (documents: Document[]): string => {
  return documents.map((document) => {
    return `<doc>\n${document.pageContent}\n</doc>`
  }).join("\n");
};

/*
{
question: "What is deep learning?"
}
*/

const documentRetrievalChain = RunnableSequence.from([
    (input) => input.question,
    retriever,
    convertDocsToString
]);

const results = await documentRetrievalChain.invoke({
  question: "What are the prerequisites for this course?"
});
console.log(results);

# Synthesizing a response

import { ChatPromptTemplate } from "@langchain/core/prompts";

const TEMPLATE_STRING = `You are an experienced researcher, 
expert at interpreting and answering questions based on provided sources.
Using the provided context, answer the user's question 
to the best of your ability using only the resources provided. 
Be verbose!

<context>

{context}

</context>

Now, answer this question using the above context:

{question}`;

const answerGenerationPrompt = ChatPromptTemplate.fromTemplate(
    TEMPLATE_STRING
);

import { RunnableMap } from "@langchain/core/runnables";

const runnableMap = RunnableMap.from({
  context: documentRetrievalChain,
  question: (input) => input.question,
});

await runnableMap.invoke({
    question: "What are the prerequisites for this course?"
})

# Augmented generation

import { ChatOpenAI } from "@langchain/openai";
import { StringOutputParser } from "@langchain/core/output_parsers";

const model = new ChatOpenAI({
    modelName: "gpt-3.5-turbo-1106"
});

const retrievalChain = RunnableSequence.from([
  {
    context: documentRetrievalChain,
    question: (input) => input.question,
  },
  answerGenerationPrompt,
  model,
  new StringOutputParser(),
]);

const answer = await retrievalChain.invoke({
  question: "What are the prerequisites for this course?"
});

console.log(answer);

const followupAnswer = await retrievalChain.invoke({
  question: "Can you list them in bullet point form?"
});

console.log(followupAnswer);

const docs = await documentRetrievalChain.invoke({
  question: "Can you list them in bullet point form?"
});

console.log(docs);























