# Keyword Search

## Setup

Load needed API keys and relevant Python libaries.

# !pip install cohere
# !pip install weaviate-client

import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

Let's start by imporing Weaviate to access the Wikipedia database.

import weaviate
auth_config = weaviate.auth.AuthApiKey(
    api_key=os.environ['WEAVIATE_API_KEY'])



client = weaviate.Client(
    url=os.environ['WEAVIATE_API_URL'],
    auth_client_secret=auth_config,
    additional_headers={
        "X-Cohere-Api-Key": os.environ['COHERE_API_KEY'],
    }
)

client.is_ready()

# Keyword Search

def keyword_search(query,
                   results_lang='en',
                   properties = ["title","url","text"],
                   num_results=3):

    where_filter = {
    "path": ["lang"],
    "operator": "Equal",
    "valueString": results_lang
    }

    response = (
        client.query.get("Articles", properties)
        .with_bm25(
            query=query
        )
        .with_where(where_filter)
        .with_limit(num_results)
        .do()
        )

    result = response['data']['Get']['Articles']
    return result

query = "What is the most viewed televised event?"
keyword_search_results = keyword_search(query)
print(keyword_search_results)

### Try modifying the search options
- Other languages to try: `en, de, fr, es, it, ja, ar, zh, ko, hi`

properties = ["text", "title", "url",
             "views", "lang"]

def print_result(result):
    """ Print results with colorful formatting """
    for i,item in enumerate(result):
        print(f'item {i}')
        for key in item.keys():
            print(f"{key}:{item.get(key)}")
            print()
        print()

print_result(keyword_search_results)

query = "What is the most viewed televised event?"
keyword_search_results = keyword_search(query, results_lang='de')
print_result(keyword_search_results)





## How to get your own API key

For this course, an API key is provided for you.  If you would like to develop projects with Cohere's API outside of this classroom, you can register for an API key [here](https://dashboard.cohere.ai/welcome/register?utm_source=partner&utm_medium=website&utm_campaign=DeeplearningAI).
