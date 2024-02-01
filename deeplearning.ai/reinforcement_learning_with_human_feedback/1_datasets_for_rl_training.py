# Lesson 2: Datasets For Reinforcement Learning Training

### Loading and exploring the datasets

"""

"Reinforcement Learning from Human Feedback" **(RLHF)** requires the following datasets:
- Preference dataset
  - Input prompt, candidate response 0, candidate response 1, choice (candidate 0 or 1)
- Prompt dataset
  - Input prompt only, no response
"""

#### Preference dataset

preference_dataset_path = "sample_preference.jsonl"

import json

preference_data = []

with open(preference_dataset_path) as f:
    for line in f:
        preference_data.append(json.loads(line))

# - Print out to explore the preference dataset

sample_1 = preference_data[0]

print(type(sample_1))


# This dictionary has four keys
print(sample_1.keys())

"""
dict_keys(['input_text', 'candidate_0', 'candidate_1', 'choice'])

"""

# - Key: 'input_test' is a prompt.

sample_1["input_text"]

"""
'I live right next to a huge university, and have been applying for a variety of
jobs with them through their faceless electronic jobs portal (the "click here to
apply for this job" type thing) for a few months. \n\nThe very first job I applied
for, I got an interview that went just so-so. But then, I never heard back (I even
looked up the number of the person who called me and called her back, left a voicemail,
never heard anything).\n\nNow, when I\'m applying for subsequent jobs - is it that same
HR person who is seeing all my applications?? Or are they forwarded to the specific departments?
\n\nI\'ve applied for five jobs there in the last four months, all the resumes and cover letters
tailored for each open position. Is this hurting my chances? I never got another interview there,
for any of the positions. [summary]: '
"""

# Try with another examples from the list, and discover that all data end the same way
preference_data[2]["input_text"][-50:]

"""
'plan something in those circumstances. [summary]: '

"""

# All our dataset ends with above and its important that they do, so that our data
# matches how the prompts is formatted to ensure the model can recognize the pattern

# - Print 'candidate_0' and 'candidate_1', these are the completions for the same prompt.

print(f"candidate_0:\n{sample_1.get('candidate_0')}\n")
print(f"candidate_1:\n{sample_1.get('candidate_1')}\n")

"""
candidate_0:
 When applying through a massive job portal, is just one HR person seeing ALL of them?

candidate_1:
 When applying to many jobs through a single university jobs portal, is just one HR person reading ALL my applications?

"""

# - Print 'choice', this is the human labeler's preference for the results completions (candidate_0 and candidate_1)

print(f"choice: {sample_1.get('choice')}")

"""
choice: 1
"""

# In this scenario, the labeller preferred 1

#### Prompt dataset

prompt_dataset_path = "sample_prompt.jsonl"

prompt_data = []

with open(prompt_dataset_path) as f:
    for line in f:
        prompt_data.append(json.loads(line))

# Check how many prompts there are in this dataset
len(prompt_data)

# **Note**: It is important that the prompts in both datasets, the preference and the prompt, come from the same distribution.

# For this lesson, all the prompts come from the same dataset of [Reddit posts](https://github.com/openai/summarize-from-feedback).

# Function to print the information in the prompt dataset with a better visualization
def print_d(d):
    for key, val in d.items():
        print(f"key:{key}\nval:{val}\n")


print_d(prompt_data[0])

"""
key:input_text
val:I noticed this the very first day! I took a picture of it to send to one of my friends who is a fellow redditor. Later when I was getting to know my suitemates, I asked them if they ever used reddit, and they showed me the stencil they used to spray that! Along with the lion which is his trademark.
 But [summary]:

"""

# Try with another prompt from the list
print_d(prompt_data[1])

"""
key:input_text
val:Nooooooo, I loved my health class! My teacher was amazing! Most days we just went outside and played and the facility allowed it because the health teacher's argument was that teens need to spend time outside everyday and he let us do that. The other days were spent inside with him teaching us how to live a healthy lifestyle. He had guest speakers come in and reach us about nutrition and our final was open book...if we even had a final.... [summary]:

"""
