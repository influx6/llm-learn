# Lesson 1: Overview

In this lesson, you will:
1. Explore the dataset of LLM prompts and responses named **chats.csv** that we’ll use throughout this course.
2. Get a fast demo overview of all the techniques showcased in greater detail in later lessons.


## Dataset

import helpers

import pandas as pd

chats = pd.read_csv("./chats.csv")

chats.head(5)

pd.set_option('display.max_colwidth', None)

chats.head(5)

## Setup and explore whylogs and langkit

import whylogs as why

why.init("whylabs_anonymous")

from langkit import llm_metrics

schema = llm_metrics.init()

result = why.log(chats,
                 name="LLM chats dataset",
                 schema=schema)

### Prompt-response relevance

from langkit import input_output

**Note**: To view the next visual, you may have to either hide the left-side menu bar or widen the notebook towards the right.

helpers.visualize_langkit_metric(
    chats,
    "response.relevance_to_prompt"
)

helpers.show_langkit_critical_queries(
    chats,
    "response.relevance_to_prompt"
)

### Data Leakage

from langkit import regexes

**Note**: To view the next visuals, you may have to either hide the left-side menu bar or widen the notebook towards the right.

helpers.visualize_langkit_metric(
    chats,
    "prompt.has_patterns"
)

helpers.visualize_langkit_metric(
    chats, 
    "response.has_patterns")

### Toxicity

from langkit import toxicity

**Note**: To view the next visuals, you may have to either hide the left-side menu bar or widen the notebook towards the right.

helpers.visualize_langkit_metric(
    chats, 
    "prompt.toxicity")

helpers.visualize_langkit_metric(
    chats, 
    "response.toxicity")

### Injections

from langkit import injections

**Note**: To view the next visual, you may have to either hide the left-side menu bar or widen the notebook towards the right.

helpers.visualize_langkit_metric(
    chats,
    "injection"
)

helpers.show_langkit_critical_queries(
    chats,
    "injection"
)

## Evaluation

helpers.evaluate_examples()

filtered_chats = chats[
    chats["response"].str.contains("Sorry")
]

filtered_chats

helpers.evaluate_examples(filtered_chats)

filtered_chats = chats[
    chats["prompt"].str.len() > 250
]

filtered_chats

helpers.evaluate_examples(filtered_chats)

























