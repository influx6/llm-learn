# Lesson 2: Using a String Template


#### Setup
Set the MakerSuite API key with the provided helper function.

import os
from utils import get_api_key
import google.generativeai as palm
from google.api_core import client_options as client_options_lib

palm.configure(
    api_key=get_api_key(),
    transport="rest",
    client_options=client_options_lib.ClientOptions(
        api_endpoint=os.getenv("GOOGLE_API_BASE"),
    )
)

#### Pick the model that generates text

models = [m for m in palm.list_models() if 'generateText' in m.supported_generation_methods]
model_bison = models[0]
model_bison

#### Helper function to call the PaLM API

from google.api_core import retry
@retry.Retry()
def generate_text(prompt,
                  model=model_bison,
                  temperature=0.0):
    return palm.generate_text(prompt=prompt,
                              model=model,
                              temperature=temperature)

#### Prompt template

1. priming: getting the LLM ready for the type of task you'll ask it to do.
2. question: the specific task.
3. decorator: how to provide or format the output.

prompt_template = """
{priming}

{question}

{decorator}

Your solution:
"""

priming_text = "You are an expert at writing clear, concise, Python code."

question = "create a doubly linked list"

#### Observe how the decorator affects the output
- In other non-coding prompt engineering tasks, it's common to use "chain-of-thought prompting" by asking the model to work through the task "step by step".
- For certain tasks like generating code, you may want to experiment with other wording that would make sense if you were asking a developer the same question.

In the code cell below, try out option 1 first, then try out option 2.

# option 1
# decorator = "Work through it step by step, and show your work. One step per line."

# option 2
decorator = "Insert comments for each line of code."

prompt = prompt_template.format(priming=priming_text,
                                question=question,
                                decorator=decorator)

#### review the prompt

print(prompt)

#### Call the API to get the completion

completion = generate_text(prompt)
print(completion.result)

#### Try another question

question = """create a very large list of random numbers in python,
and then write code to sort that list"""

prompt = prompt_template.format(priming=priming_text,
                                question=question,
                                decorator=decorator)

print(prompt)

completion = generate_text(prompt)
print(completion.result)

#### Try out the generated code
- Debug it as needed.  For instance, you may need to import `random` in order to use the `random.randint()` function.

# copy-paste some of the generated code that generates random numbers
random_numbers = [random.randint(0, 100) for _ in range(100000)]
print(random_numbers)
