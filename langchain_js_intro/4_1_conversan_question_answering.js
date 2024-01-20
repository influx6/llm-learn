# Lesson 5: Conversational Question answering

![](./images/rag_diagram.png)

# Retrieval chain from previous lesson

Let's split and load our documents into a vector store and create a retriever. Then we will convert its output to a string.

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

import { RunnableSequence } from "@langchain/core/runnables";
import { Document } from "@langchain/core/documents";

const convertDocsToString = (documents: Document[]): string => {
  return documents.map((document) => {
    return `<doc>\n${document.pageContent}\n</doc>`
  }).join("\n");
};

const documentRetrievalChain = RunnableSequence.from([
    (input) => input.question,
    retriever,
    convertDocsToString
]);

now that we have a retriever, lets build a retriever chain.

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

# Adding history

import { MessagesPlaceholder } from "@langchain/core/prompts";

const REPHRASE_QUESTION_SYSTEM_TEMPLATE = 
  `Given the following conversation and a follow up question, 
rephrase the follow up question to be a standalone question.`;

const rephraseQuestionChainPrompt = ChatPromptTemplate.fromMessages([
  ["system", REPHRASE_QUESTION_SYSTEM_TEMPLATE],
  new MessagesPlaceholder("history"),
  [
    "human", 
    "Rephrase the following question as a standalone question:\n{question}"
  ],
]);

const rephraseQuestionChain = RunnableSequence.from([
      rephraseQuestionChainPrompt,
      new ChatOpenAI({ temperature: 0.1, modelName: "gpt-3.5-turbo-1106" }),
      new StringOutputParser(),
])

import { HumanMessage, AIMessage } from "@langchain/core/messages";

const originalQuestion = "What are the prerequisites for this course?";

const originalAnswer = await retrievalChain.invoke({
  question: originalQuestion
});

console.log(originalAnswer);

const chatHistory = [
  new HumanMessage(originalQuestion),
  new AIMessage(originalAnswer),
];

await rephraseQuestionChain.invoke({
  question: "Can you list them in bullet point form?",
  history: chatHistory,
});

# Putting it all together

const convertDocsToString = (documents: Document[]): string => {
  return documents.map((document) => `<doc>\n${document.pageContent}\n</doc>`).join("\n");
};

const documentRetrievalChain = RunnableSequence.from([
  (input) => input.standalone_question,
  retriever,
  convertDocsToString,
]);

const ANSWER_CHAIN_SYSTEM_TEMPLATE = `You are an experienced researcher, 
expert at interpreting and answering questions based on provided sources.
Using the below provided context and chat history, 
answer the user's question to the best of 
your ability 
using only the resources provided. Be verbose!

<context>
{context}
</context>`;

const answerGenerationChainPrompt = ChatPromptTemplate.fromMessages([
  ["system", ANSWER_CHAIN_SYSTEM_TEMPLATE],
  new MessagesPlaceholder("history"),
  [
    "human", 
    "Now, answer this question using the previous context and chat history:\n{standalone_question}"
  ]
]);

import { HumanMessage, AIMessage } from "@langchain/core/messages";
await answerGenerationChainPrompt.formatMessages({
  context: "fake retrieved content",
  standalone_question: "Why is the sky blue?",
  history: [
    new HumanMessage("How are you?"),
    new AIMessage("Fine, thank you!")
  ]
});

import { RunnablePassthrough } from "@langchain/core/runnables";

const conversationalRetrievalChain = RunnableSequence.from([
  RunnablePassthrough.assign({
    standalone_question: rephraseQuestionChain,
  }),
  RunnablePassthrough.assign({
    context: documentRetrievalChain,
  }),
  answerGenerationChainPrompt,
  new ChatOpenAI({ modelName: "gpt-3.5-turbo" }),
  new StringOutputParser(),
]);

import { RunnableWithMessageHistory } from "@langchain/core/runnables";
import { ChatMessageHistory } from "langchain/stores/message/in_memory";

const messageHistory = new ChatMessageHistory();

const finalRetrievalChain = new RunnableWithMessageHistory({
  runnable: conversationalRetrievalChain,
  getMessageHistory: (_sessionId) => messageHistory,
  historyMessagesKey: "history",
  inputMessagesKey: "question",
});

const originalQuestion = "What are the prerequisites for this course?";

const originalAnswer = await finalRetrievalChain.invoke({
  question: originalQuestion,
}, {
  configurable: { sessionId: "test" }
});

const finalResult = await finalRetrievalChain.invoke({
  question: "Can you list them in bullet point form?",
}, {
  configurable: { sessionId: "test" }
});

console.log(finalResult);

https://smith.langchain.com/public/fca11abd-c0ec-456f-800e-6edfaf6bcf68/r























