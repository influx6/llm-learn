# Lesson 2: RAG Triad of metrics

import warnings
warnings.filterwarnings('ignore')

import utils

import os
import openai
openai.api_key = utils.get_openai_api_key()

from trulens_eval import Tru

tru = Tru()
tru.reset_database()

from llama_index import SimpleDirectoryReader

documents = SimpleDirectoryReader(
    input_files=["./eBook-How-to-Build-a-Career-in-AI.pdf"]
).load_data()

from llama_index import Document

document = Document(text="\n\n".\
                    join([doc.text for doc in documents]))

from utils import build_sentence_window_index

from llama_index.llms import OpenAI

llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)

sentence_index = build_sentence_window_index(
    document,
    llm,
    embed_model="local:BAAI/bge-small-en-v1.5",
    save_dir="sentence_index"
)

from utils import get_sentence_window_query_engine

sentence_window_engine = \
get_sentence_window_query_engine(sentence_index)

output = sentence_window_engine.query(
    "How do you create your AI portfolio?")
output.response

## Feedback functions

import nest_asyncio

nest_asyncio.apply()

from trulens_eval import OpenAI as fOpenAI


### Feedback function provides a score after reviewing an LLM apps inputs, outputs and intermediate results
### Such feedback functions are: the answer relevance, context relevance and groundness functions in trulens used to evaluate
### the llm outputs

provider = fOpenAI() # this is the provider or model (for now OpenAI) model to use to do feedback but you can use other things even non-llm providers

## 1. answer relevance - how useful is the final response to the question/query

from trulens_eval import Feedback

f_qa_relevance = Feedback(
    provider.relevance_with_cot_reasons,
    name="Answer Relevance"
).on_input_output()

### 2. Context Relevance - how good the reteival process is, with the query, assess each piece of retreived context and check how relevant are they to the query/input.

from trulens_eval import TruLlama

context_selection = TruLlama.select_source_nodes().node.text

import numpy as np

f_qs_relevance = (
    Feedback(provider.qs_relevance,
             name="Context Relevance")
    .on_input()
    .on(context_selection)
    .aggregate(np.mean)
)

import numpy as np

f_qs_relevance = (
    Feedback(provider.qs_relevance_with_cot_reasons,
             name="Context Relevance")
    .on_input()
    .on(context_selection)
    .aggregate(np.mean)
)

### 3. Groundedness - checks the chain of thought reasoning, the context and output, it checks all intermediate outputs and final outputs for groundedness score

from trulens_eval.feedback import Groundedness

grounded = Groundedness(groundedness_provider=provider)

f_groundedness = (
    Feedback(grounded.groundedness_measure_with_cot_reasons,
             name="Groundedness"
            )
    .on(context_selection)
    .on_output()
    .aggregate(grounded.grounded_statements_aggregator)
)

## Evaluation of the RAG application

from trulens_eval import TruLlama
from trulens_eval import FeedbackMode

tru_recorder = TruLlama(
    sentence_window_engine,
    app_id="App_1",
    feedbacks=[
        f_qa_relevance,
        f_qs_relevance,
        f_groundedness
    ]
)

eval_questions = []
with open('eval_questions.txt', 'r') as file:
    for line in file:
        # Remove newline character and convert to integer
        item = line.strip()
        eval_questions.append(item)

eval_questions

eval_questions.append("How can I be successful in AI?")

eval_questions

for question in eval_questions:
    with tru_recorder as recording:
        sentence_window_engine.query(question)

records, feedback = tru.get_records_and_feedback(app_ids=[])
records.head()

import pandas as pd

pd.set_option("display.max_colwidth", None)
records[["input", "output"] + feedback]

tru.get_leaderboard(app_ids=[])

tru.run_dashboard()