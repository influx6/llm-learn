# Lesson 4: Evaluate the Tuned Model

#### Project environment setup

- Install Tensorboard (if running locally)
```Python
!pip install tensorboard
```

### Explore results with Tensorboard

%load_ext tensorboard

port = %env PORT1
%tensorboard --logdir reward-logs --port $port --bind_all

# Look at what this directory has
%ls reward-logs

port = %env PORT2
%tensorboard --logdir reinforcer-logs --port $port --bind_all

port = %env PORT3
%tensorboard --logdir reinforcer-fulldata-logs --port $port --bind_all

- The dictionary of 'parameter_values' defined in the previous lesson

parameter_values={
        "preference_dataset": \
    "gs://vertex-ai/generative-ai/rlhf/text_small/summarize_from_feedback_tfds/comparisons/train/*.jsonl",
        "prompt_dataset": \
    "gs://vertex-ai/generative-ai/rlhf/text_small/reddit_tfds/train/*.jsonl",
        "eval_dataset": \
    "gs://vertex-ai/generative-ai/rlhf/text_small/reddit_tfds/val/*.jsonl",
        "large_model_reference": "llama-2-7b",
        "reward_model_train_steps": 1410,
        "reinforcement_learning_train_steps": 320,
        "reward_model_learning_rate_multiplier": 1.0,
        "reinforcement_learning_rate_multiplier": 1.0,
        "kl_coeff": 0.1,
        "instruction":\
    "Summarize in less than 50 words"}

**Note:** Here, we are using "text_small" for our datasets for learning purposes. However for the results that we're evaluating in this lesson, the team used the full dataset with the following hyperparameters:

```Python
parameter_values={
        "preference_dataset": \
    "gs://vertex-ai/generative-ai/rlhf/text/summarize_from_feedback_tfds/comparisons/train/*.jsonl",
        "prompt_dataset": \
    "gs://vertex-ai/generative-ai/rlhf/text/reddit_tfds/train/*.jsonl",
        "eval_dataset": \
    "gs://vertex-ai/generative-ai/rlhf/text/reddit_tfds/val/*.jsonl",
        "large_model_reference": "llama-2-7b",
        "reward_model_train_steps": 10000,
        "reinforcement_learning_train_steps": 10000,
        "reward_model_learning_rate_multiplier": 1.0,
        "reinforcement_learning_rate_multiplier": 0.2,
        "kl_coeff": 0.1,
        "instruction":\
    "Summarize in less than 50 words"}
```

### Evaluate using the tuned and untuned model

import json

eval_tuned_path = 'eval_results_tuned.jsonl'

eval_data_tuned = []

with open(eval_tuned_path) as f:
    for line in f:
        eval_data_tuned.append(json.loads(line))

# Import for printing purposes
from utils import print_d

# Look at the result produced by the tuned model
print_d(eval_data_tuned[0])

eval_untuned_path = 'eval_results_untuned.jsonl'

eval_data_untuned = []

with open(eval_untuned_path) as f:
    for line in f:
        eval_data_untuned.append(json.loads(line))

# Look at the result produced by the untuned model
print_d(eval_data_untuned[0])

### Explore the results side by side in a dataframe

# Extract all the prompts
prompts = [sample['inputs']['inputs_pretokenized']
           for sample in eval_data_tuned]

# Completions from the untuned model
untuned_completions = [sample['prediction']
                       for sample in eval_data_untuned]

# Completions from the tuned model
tuned_completions = [sample['prediction']
                     for sample in eval_data_tuned]

- Now putting all together in one big dataframe

import pandas as pd

results = pd.DataFrame(
    data={'prompt': prompts,
          'base_model':untuned_completions,
          'tuned_model': tuned_completions})

pd.set_option('display.max_colwidth', None)

# Print the results
results
