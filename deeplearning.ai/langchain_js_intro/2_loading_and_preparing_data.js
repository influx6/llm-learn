# Lesson 2: Loading and preparing data

# Loading

import "dotenv/config";

import { GithubRepoLoader } from "langchain/document_loaders/web/github";
// Peer dependency, used to support .gitignore syntax
import ignore from "ignore";

// Will not include anything under "ignorePaths"
const loader = new GithubRepoLoader(
  "https://github.com/langchain-ai/langchainjs",
  { recursive: false, ignorePaths: ["*.md", "yarn.lock"] }
);

const docs = await loader.load();

console.log(docs.slice(0, 3));

// Peer dependency
import * as parse from "pdf-parse";
import { PDFLoader } from "langchain/document_loaders/fs/pdf";

const loader = new PDFLoader("./data/MachineLearning-Lecture01.pdf");

const rawCS229Docs = await loader.load();

console.log(rawCS229Docs.slice(0, 5));

# Splitting

import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

const splitter = RecursiveCharacterTextSplitter.fromLanguage("js", {
  chunkSize: 32,
  chunkOverlap: 0,
});

const code = `function helloWorld() {
console.log("Hello, World!");
}
// Call the function
helloWorld();`;

await splitter.splitText(code);

import { CharacterTextSplitter } from "langchain/text_splitter";

const splitter = new CharacterTextSplitter({
  chunkSize: 32,
  chunkOverlap: 0,
  separator: " "
});

await splitter.splitText(code);

const splitter = RecursiveCharacterTextSplitter.fromLanguage("js", {
  chunkSize: 64,
  chunkOverlap: 32,
});

await splitter.splitText(code);

const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 512,
  chunkOverlap: 64,
});

const splitDocs = await splitter.splitDocuments(rawCS229Docs);

console.log(splitDocs.slice(0, 5));

























