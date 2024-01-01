## Lesson 4: Applications of Embeddings

#### Project environment setup

- Load credentials and relevant Python Libraries

from utils import authenticate
credentials, PROJECT_ID = authenticate()

REGION = 'us-central1'

import vertexai
vertexai.init(project=PROJECT_ID,
              location=REGION,
              credentials = credentials)

#### Load Stack Overflow questions and answers from BigQuery
# - BigQuery is Google Cloud's serverless data warehouse.
# - We'll get the first 500 posts (questions and answers) for each programming language: Python, HTML, R, and CSS.

from google.cloud import bigquery
import pandas as pd

def run_bq_query(sql):

    # Create BQ client
    bq_client = bigquery.Client(project = PROJECT_ID,
                                credentials = credentials)

    # Try dry run before executing query to catch any errors
    job_config = bigquery.QueryJobConfig(dry_run=True,
                                         use_query_cache=False)
    bq_client.query(sql, job_config=job_config)

    # If dry run succeeds without errors, proceed to run query
    job_config = bigquery.QueryJobConfig()
    client_result = bq_client.query(sql,
                                    job_config=job_config)

    job_id = client_result.job_id

    # Wait for query/job to finish running. then get & return data frame
    df = client_result.result().to_arrow().to_pandas()
    print(f"Finished job_id: {job_id}")
    return df

# define list of programming language tags we want to query

language_list = ["python", "html", "r", "css"]

so_df = pd.DataFrame()

for language in language_list:

    print(f"generating {language} dataframe")

    query = f"""
    SELECT
        CONCAT(q.title, q.body) as input_text,
        a.body AS output_text
    FROM
        `bigquery-public-data.stackoverflow.posts_questions` q
    JOIN
        `bigquery-public-data.stackoverflow.posts_answers` a
    ON
        q.accepted_answer_id = a.id
    WHERE
        q.accepted_answer_id IS NOT NULL AND
        REGEXP_CONTAINS(q.tags, "{language}") AND
        a.creation_date >= "2020-01-01"
    LIMIT
        500
    """


    language_df = run_bq_query(query)
    language_df["category"] = language
    so_df = pd.concat([so_df, language_df],
                      ignore_index = True)

- You can reuse the above code to run your own queries if you are using Google Cloud's BigQuery service.
- In this classroom, if you run into any issues, you can load the same data from a csv file.

# Run this cell if you get any errors or you don't want to wait for the query to be completed
# so_df = pd.read_csv('so_database_app.csv')

so_df

#### Generate text embeddings
- To generate embeddings for a dataset of texts, we'll need to group the sentences together in batches and send batches of texts to the model.
- The API currently can take batches of up to 5 pieces of text per API call.

from vertexai.language_models import TextEmbeddingModel

model = TextEmbeddingModel.from_pretrained(
    "textembedding-gecko@001")

import time
import numpy as np

# Generator function to yield batches of sentences

def generate_batches(sentences, batch_size = 5):
    for i in range(0, len(sentences), batch_size):
        yield sentences[i : i + batch_size]

so_questions = so_df[0:200].input_text.tolist()
batches = generate_batches(sentences = so_questions)

batch = next(batches)
len(batch)

#### Get embeddings on a batch of data
- This helper function calls `model.get_embeddings()` on the batch of data, and returns a list containing the embeddings for each text in that batch.

def encode_texts_to_embeddings(sentences):
    try:
        embeddings = model.get_embeddings(sentences)
        return [embedding.values for embedding in embeddings]
    except Exception:
        return [None for _ in range(len(sentences))]

batch_embeddings = encode_texts_to_embeddings(batch)

f"{len(batch_embeddings)} embeddings of size \
{len(batch_embeddings[0])}"

#### Code for getting data on an entire data set
- Most API services have rate limits, so we've provided a helper function (in utils.py) that you could use to wait in-between API calls.
- If the code was not designed to wait in-between API calls, you may not receive embeddings for all batches of text.
- This particular service can handle 20 calls per minute.  In calls per second, that's 20 calls divided by 60 seconds, or `20/60`.

```Python
from utils import encode_text_to_embedding_batched

so_questions = so_df.input_text.tolist()
question_embeddings = encode_text_to_embedding_batched(
                            sentences=so_questions,
                            api_calls_per_second = 20/60,
                            batch_size = 5)
```

In order to handle limits of this classroom environment, we're not going to run this code to embed all of the data. But you can adapt this code for your own projects and datasets.

#### Load the data from file
- We'll load the stack overflow questions, answers, and category labels (Python, HTML, R, CSS) from a .csv file.
- We'll load the embeddings of the questions (which we've precomputed with batched calls to `model.get_embeddings()`), from a pickle file.

so_df = pd.read_csv('so_database_app.csv')
so_df.head()

import pickle

with open('question_embeddings_app.pkl', 'rb') as file:
    question_embeddings = pickle.load(file)

print("Shape: " + str(question_embeddings.shape))
print(question_embeddings)

#### Cluster the embeddings of the Stack Overflow questions

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

clustering_dataset = question_embeddings[:1000]

n_clusters = 2
kmeans = KMeans(n_clusters=n_clusters,
                random_state=0,
                n_init = 'auto').fit(clustering_dataset)

kmeans_labels = kmeans.labels_

PCA_model = PCA(n_components=2)
PCA_model.fit(clustering_dataset)
new_values = PCA_model.transform(clustering_dataset)

import matplotlib.pyplot as plt
import mplcursors
%matplotlib ipympl

from utils import clusters_2D
clusters_2D(x_values = new_values[:,0], y_values = new_values[:,1],
            labels = so_df[:1000], kmeans_labels = kmeans_labels)

- Clustering is able to identify two distinct clusters of HTML or Python related questions, without being given the category labels (HTML or Python).

## Anomaly / Outlier detection

- We can add an anomalous piece of text and check if the outlier (anomaly) detection algorithm (Isolation Forest) can identify it as an outlier (anomaly), based on its embedding.

from sklearn.ensemble import IsolationForest

input_text = """I am making cookies but don't
                remember the correct ingredient proportions.
                I have been unable to find
                anything on the web."""

emb = model.get_embeddings([input_text])[0].values

embeddings_l = question_embeddings.tolist()
embeddings_l.append(emb)

embeddings_array = np.array(embeddings_l)

print("Shape: " + str(embeddings_array.shape))
print(embeddings_array)

# Add the outlier text to the end of the stack overflow dataframe
so_df = pd.read_csv('so_database_app.csv')
new_row = pd.Series([input_text, None, "baking"],
                    index=so_df.columns)
so_df.loc[len(so_df)+1] = new_row
so_df.tail()

#### Use Isolation Forest to identify potential outliers

- `IsolationForest` classifier will predict `-1` for potential outliers, and `1` for non-outliers.
- You can inspect the rows that were predicted to be potential outliers and verify that the question about baking is predicted to be an outlier.

clf = IsolationForest(contamination=0.005,
                      random_state = 2)

preds = clf.fit_predict(embeddings_array)

print(f"{len(preds)} predictions. Set of possible values: {set(preds)}")

so_df.loc[preds == -1]

#### Remove the outlier about baking

so_df = so_df.drop(so_df.index[-1])

so_df

## Classification
- Train a random forest model to classify the category of a Stack Overflow question (as either Python, R, HTML or CSS).

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# re-load the dataset from file
so_df = pd.read_csv('so_database_app.csv')
X = question_embeddings
X.shape

y = so_df['category'].values
y.shape

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size = 0.2,
                                                    random_state = 2)

clf = RandomForestClassifier(n_estimators=200)

clf.fit(X_train, y_train)

#### You can check the predictions on a few questions from the test set

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred) # compute accuracy
print("Accuracy:", accuracy)

#### Try out the classifier on some questions

# choose a number between 0 and 1999
i = 2
label = so_df.loc[i,'category']
question = so_df.loc[i,'input_text']

# get the embedding of this question and predict its category
question_embedding = model.get_embeddings([question])[0].values
pred = clf.predict([question_embedding])

print(f"For question {i}, the prediction is `{pred[0]}`")
print(f"The actual label is `{label}`")
print("The question text is:")
print("-"*50)
print(question)
