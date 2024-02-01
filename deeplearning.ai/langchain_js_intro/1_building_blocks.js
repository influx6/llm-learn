# Lesson 1: Building blocks

Welcome to this course! Just a reminder that LLMs may not always return the same text. As a result, the outputs you see when running the notebook may vary somewhat from those you see in the film. This applies to all of the notebooks.
Enjoy the course!

## Language model

1. Text LLMs: string -> string
2. Chat models: list of messages -> single message output

import "dotenv/config";

import { ChatOpenAI } from "@langchain/openai";
import { HumanMessage } from "@langchain/core/messages";

const model = new ChatOpenAI({
    modelName: "gpt-3.5-turbo-1106"
});

await model.invoke([
    new HumanMessage("Tell me a joke.")
]);

## Prompt template

import { ChatPromptTemplate } from "@langchain/core/prompts";

const prompt = ChatPromptTemplate.fromTemplate(
    `What are three good names for a company that makes {product}?`
)

await prompt.format({
    product: "colorful socks"
});

await prompt.formatMessages({
    product: "colorful socks"
});

import { 
  SystemMessagePromptTemplate, 
  HumanMessagePromptTemplate 
} from "@langchain/core/prompts";

const promptFromMessages = ChatPromptTemplate.fromMessages([
  SystemMessagePromptTemplate.fromTemplate(
    "You are an expert at picking company names."
  ),
  HumanMessagePromptTemplate.fromTemplate(
    "What are three good names for a company that makes {product}?"
  )
]);

await promptFromMessages.formatMessages({
    product: "shiny objects"
});

const promptFromMessages = ChatPromptTemplate.fromMessages([
  ["system", "You are an expert at picking company names."],
  ["human", "What are three good names for a company that makes {product}?"]
]);

await promptFromMessages.formatMessages({
  product: "shiny objects"
});

# LangChain Expression Language (LCEL)

const chain = prompt.pipe(model);

await chain.invoke({
    product: "colorful socks"
});

## Output parser

import { StringOutputParser } from "@langchain/core/output_parsers";

const outputParser = new StringOutputParser();

const nameGenerationChain = prompt.pipe(model).pipe(outputParser);

await nameGenerationChain.invoke({
    product: "fancy cookies"
});

https://smith.langchain.com/public/9fd7689f-23c6-4ea1-8a76-fea1b5f8d5db/r

import { RunnableSequence } from "@langchain/core/runnables";

const nameGenerationChain = RunnableSequence.from([
    prompt,
    model,
    outputParser
])

await nameGenerationChain.invoke({
    product: "fancy cookies"
});

## Streaming

const stream = await nameGenerationChain.stream({
  product: "really cool robots",
});

for await (const chunk of stream) {
    console.log(chunk);
}

## Batch

const inputs = [
    { product: "large calculators" },
    { product: "alpaca wool sweaters" }
];

await nameGenerationChain.batch(inputs);





















