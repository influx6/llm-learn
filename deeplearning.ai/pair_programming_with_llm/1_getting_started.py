# Lesson 1: Getting Started with PaLM

#### Setup
Set the MakerSuite API key with the provided helper function.

from utils import get_api_key

In this classroom, we've installed the relevant libraries for you.

If you wanted to use the PaLM API on your own machine, you would first install the library:
```Python
!pip install -q google.generativeai
```
The optional flag `-q` installs "quietly" without printing out details of the installation.




######

import os
import google.generativeai as palm
from google.api_core import client_options as client_options_lib

palm.configure(
    api_key=get_api_key(),
    transport="rest",
    client_options=client_options_lib.ClientOptions(
        api_endpoint=os.getenv("GOOGLE_API_BASE"),
    )
)

### Explore the available models

for m in palm.list_models():
    print(f"name: {m.name}")
    print(f"description: {m.description}")
    print(f"generation methods:{m.supported_generation_methods}\n")

#### Filter models by their supported generation methods
- `generateText` is currently recommended for coding-related prompts.
- `generateMessage` is optimized for multi-turn chats (dialogues) with an LLM.

models = [m for m in palm.list_models()
          if 'generateText'
          in m.supported_generation_methods]
models

model_bison = models[0]
model_bison

#### helper function to generate text

- The `@retry` decorator helps you to retry the API call if it fails.
- We set the temperature to 0.0 so that the model returns the same output (completion) if given the same input (the prompt).

from google.api_core import retry
@retry.Retry()
def generate_text(prompt,
                  model=model_bison,
                  temperature=0.0):
    return palm.generate_text(prompt=prompt,
                              model=model,
                              temperature=temperature)

#### Ask the LLM how to write some code



prompt = "Show me how to iterate across a list in Python."

completion = generate_text(prompt)

print(completion.result)

- Tip: The words "show me" tends to encourage the PaLM LLM to give more details and explanations compared to if you were to ask "write code to ..."

prompt = "write code to iterate across a list in Python"

completion = generate_text(prompt)
print(completion.result)

#### Try out the code
- Try copy-pasting some of the generated code and running it in the notebook.
- Remember to test out the LLM-generated code and debug it make sure it works as intended.

# paste the LLM's code here




#### Try asking your own coding question

# Modify the prompt with your own question
prompt = "Show me how to [...]"

completion = generate_text(prompt)



#### Note about the API key
We've provided an API key for this classroom.  If you would like your own API key for your own projects, you can get one at [developers.generativeai.google](https://developers.generativeai.google/)