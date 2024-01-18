# Lesson 3: Data leakage and toxicity

## Setup

import pandas as pd

pd.set_option('display.max_colwidth', None)

import whylogs as why

import helpers

chats = pd.read_csv("./chats.csv")

chats[10:11]

## Data leakage 

### 1. Detect Patterns

from langkit import regexes

**Note**: To view the next visuals, you may have to either hide the left-side menu bar or widen the notebook towards the right.

helpers.visualize_langkit_metric(
    chats, 
    "prompt.has_patterns"
)

helpers.visualize_langkit_metric(
    chats, 
    "response.has_patterns"
)

helpers.show_langkit_critical_queries(
    chats, 
    "response.has_patterns"
)

from whylogs.experimental.core.udf_schema import udf_schema

annotated_chats, _ = udf_schema().apply_udfs(chats)

annotated_chats.head(5)

annotated_chats[(annotated_chats["prompt.has_patterns"].notnull()) |
                  (annotated_chats["response.has_patterns"].notnull())]

**Note**: To view the next visual, you may have to either hide the left-side menu bar or widen the notebook towards the right.

helpers.evaluate_examples(
  annotated_chats[(annotated_chats["prompt.has_patterns"].notnull()) |
                  (annotated_chats["response.has_patterns"].notnull())] ,
  scope="leakage")

### 2. Entity recognition

from span_marker import SpanMarkerModel

entity_model = SpanMarkerModel.from_pretrained(
    "tomaarsen/span-marker-bert-tiny-fewnerd-coarse-super"
)

entity_model.predict(
    "Write an funny email subject to Bill Gates that\
    describes a confidential product called Modelizer 900."
)

leakage_entities = ["person", "product","organization"]

from whylogs.experimental.core.udf_schema import register_dataset_udf

@register_dataset_udf(["prompt"],"prompt.entity_leakage")
def entity_leakage(text):
    entity_counts = []
    for _, row in text.iterrows():
        entity_counts.append(
            next((entity["label"] for entity in \
                entity_model.predict(row["prompt"]) if\
                entity["label"] in leakage_entities and \
                entity["score"] > 0.25), None
            )
        )
    return entity_counts

entity_leakage(chats.head(5))

@register_dataset_udf(["response"],"response.entity_leakage")
def entity_leakage(text):
    entity_counts = []
    for _, row in text.iterrows():
        entity_counts.append(
            next((entity["label"] for entity in \
                entity_model.predict(row["response"]) if\
                entity["label"] in leakage_entities and \
                entity["score"] > 0.25), None
            )
        )
    return entity_counts

annotated_chats, _ = udf_schema().apply_udfs(chats)

helpers.show_langkit_critical_queries(
    chats, 
    "prompt.entity_leakage")

annotated_chats[(annotated_chats["prompt.has_patterns"].notnull()) |
                  (annotated_chats["response.has_patterns"].notnull()) | 
                  (annotated_chats["prompt.entity_leakage"].notnull()) |
                  (annotated_chats["response.entity_leakage"].notnull())
]

**Note**: To view the next visual, you may have to either hide the left-side menu bar or widen the notebook towards the right.

helpers.evaluate_examples(
  annotated_chats[(annotated_chats["prompt.has_patterns"].notnull()) |
                  (annotated_chats["response.has_patterns"].notnull()) | 
                  (annotated_chats["prompt.entity_leakage"].notnull()) |
                  (annotated_chats["response.entity_leakage"].notnull())],
  scope="leakage")

## Toxicity

from transformers import pipeline

toxigen_hatebert = pipeline("text-classification", 
                            model="tomh/toxigen_hatebert", 
                            tokenizer="bert-base-cased")

toxigen_hatebert(["Something non-toxic",
                  "A benign sentence, despite mentioning women."])

@register_dataset_udf(["prompt"],"prompt.implicit_toxicity")
def implicit_toxicity(text):
    return [int(result["label"][-1]) for result in 
            toxigen_hatebert(text["prompt"].to_list())]

helpers.show_langkit_critical_queries(
    annotated_chats, 
    "prompt.implicit_toxicity")

