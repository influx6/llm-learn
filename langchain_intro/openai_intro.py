import os
import typing
import openai
import logging

from dotenv import load_dotenv, find_dotenv

logging.basicConfig(level=logging.DEBUG)

LOGGER = logging.getLogger(__file__)

_ = load_dotenv(find_dotenv())

openai.api_key = os.environ["OPENAI_API_KEY"]

# account for deprecation of LLM model
import datetime
# Get the current date
current_date = datetime.datetime.now().date()

# Define the date after which the model should be set to "gpt-3.5-turbo"
target_date = datetime.date(2024, 6, 12)

# Set the model variable based on the current date
if current_date > target_date:
    llm_model = "gpt-3.5-turbo"
else:
    llm_model = "gpt-3.5-turbo-0301"

def get_completion(prompt, model=llm_model):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, 
    )
    return response.choices

def get_completion_response(response_list: typing.List) -> str:
    return response_list[0].message["content"]

response_list = get_completion("what is 1+1?")
LOGGER.info(f"ChatGPT responded with: {get_completion_response(response_list)}")


customer_email = """
Arrr, I be fuming that me blender lid \
flew off and splattered me kitchen walls \
with smoothie! And to make matters worse,\
the warranty don't cover the cost of \
cleaning up me kitchen. I need yer help \
right now, matey!
"""

style = """American English \
in a calm and respectful tone
"""

prompt = f"""Translate the text \
that is delimited by triple backticks 
into a style that is {style}.
text: ```{customer_email}```
"""

response = get_completion(prompt)
LOGGER.info(f"ChatGPT responded with: {get_completion_response(response_list)}")


