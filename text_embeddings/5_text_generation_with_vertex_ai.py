## Lesson 5: Text Generation with Vertex AI

#### Project environment setup

# - Load credentials and relevant Python Libraries

from utils import authenticate

credentials, PROJECT_ID = authenticate()

REGION = "us-central1"

### Prompt the model
# - We'll import a language model that has been trained to handle a variety of natural language tasks, `text-bison@001`.
# - For multi-turn dialogue with a language model, you can use, `chat-bison@001`.

import vertexai

vertexai.init(project=PROJECT_ID, location=REGION, credentials=credentials)

from vertexai.language_models import TextGenerationModel

generation_model = TextGenerationModel.from_pretrained("text-bison@001")

#### Question Answering
# - You can ask an open-ended question to the language model.

prompt = "I'm a high school student. \
Recommend me a programming activity to improve my skills."

print(generation_model.predict(prompt=prompt).text)

"""
* **Write a program to solve a problem you're interested in.** This could be anything from a game to a tool to help you with your studies. The important thing is that you're interested in the problem and that you're motivated to solve it.
* **Take a programming course.** There are many online and offline courses available, so you can find one that fits your schedule and learning style.
* **Join a programming community.** There are many online and offline communities where you can connect with other programmers and learn from each other.
* **Read programming books and articles.** There is a
"""

#### Classify and elaborate
# - For more predictability of the language model's response, you can also ask the language model to choose among a list of answers and then elaborate on its answer.

prompt = """I'm a high school student. \
Which of these activities do you suggest and why:
a) learn Python
b) learn Javascript
c) learn Fortran
"""

print(generation_model.predict(prompt=prompt).text)

"""
I would suggest learning Python. Python is a general-purpose programming language that is easy to
learn and has a wide range of applications. It is used in a variety of fields,
including web development, data science, and machine learning. Python is also a
popular language for beginners, as it has a large community of support and resources available.
"""

#### Extract information and format it as a table

prompt = """ A bright and promising wildlife biologist \
named Jesse Plank (Amara Patel) is determined to make her \
mark on the world.
Jesse moves to Texas for what she believes is her dream job,
only to discover a dark secret that will make \
her question everything.
In the new lab she quickly befriends the outgoing \
lab tech named Maya Jones (Chloe Nguyen),
and the lab director Sam Porter (Fredrik Johansson).
Together the trio work long hours on their research \
in a hope to change the world for good.
Along the way they meet the comical \
Brenna Ode (Eleanor Garcia) who is a marketing lead \
at the research institute,
and marine biologist Siri Teller (Freya Johansson).

Extract the characters, their jobs \
and the actors who played them from the above message as a table
"""

response = generation_model.predict(prompt=prompt)

print(response.text)

"""
| Character | Job | Actor |
|---|---|---|
| Jesse Plank | Wildlife Biologist | Amara Patel |
| Maya Jones | Lab Tech | Chloe Nguyen |
| Sam Porter | Lab Director | Fredrik Johansson |
| Brenna Ode | Marketing Lead | Eleanor Garcia |
| Siri Teller | Marine Biologist | Freya Johansson |
"""

"""
 - You can copy-paste the text into a markdown cell to see if it displays a table.

| Character | Job | Actor |
|---|---|---|
| Jesse Plank | Wildlife Biologist | Amara Patel |
| Maya Jones | Lab Tech | Chloe Nguyen |
| Sam Porter | Lab Director | Fredrik Johansson |
| Brenna Ode | Marketing Lead | Eleanor Garcia |
| Siri Teller | Marine Biologist | Freya Johansson |
"""

### Adjusting Creativity/Randomness
# - You can control the behavior of the language model's decoding strategy by adjusting the temperature, top-k, and top-n parameters.
# - For tasks for which you want the model to consistently output the same result for the same input, (such as classification or information extraction), set temperature to zero.
# - For tasks where you desire more creativity, such as brainstorming, summarization, choose a higher temperature (up to 1).

temperature = 0.0

prompt = "Complete the sentence: \
As I prepared the picture frame, \
I reached into my toolkit to fetch my:"

response = generation_model.predict(
    prompt=prompt,
    temperature=temperature,
)

print(f"[temperature = {temperature}]")
print(response.text)

"""
[temperature = 0.0]
As I prepared the picture frame, I reached into my toolkit to fetch my hammer.
"""

temperature = 1.0

response = generation_model.predict(
    prompt=prompt,
    temperature=temperature,
)

print(f"[temperature = {temperature}]")
print(response.text)

"""
[temperature = 1.0]
As I prepared the picture frame, I reached into my toolkit to fetch my hammer. A hammer is a tool used to pound nails into wood. It is often used in woodworking and construction.
"""


#### Top P
# - Top p: sample the minimum set of tokens whose probabilities add up to probability `p` or greater.
# - The default value for `top_p` is `0.95`.
# - If you want to adjust `top_p` and `top_k` and see different results, remember to set `temperature` to be greater than zero, otherwise the model will always choose the token with the highest probability.

top_p = 0.2

prompt = "Write an advertisement for jackets \
that involves blue elephants and avocados."

response = generation_model.predict(
    prompt=prompt,
    temperature=0.9,
    top_p=top_p,
)

print(f"[top_p = {top_p}]")
print(response.text)

"""
[top_p = 0.2]
**Introducing the new Blue Elephant Avocado Jacket!**

This jacket is the perfect way to show off your unique style. With its bright blue color and fun avocado print, you'll be sure to turn heads wherever you go.

The Blue Elephant Avocado Jacket is made from high-quality materials that will keep you warm and comfortable all day long. It's also water-resistant, so you can wear it on rainy days without having to worry about getting wet.

Whether you're running errands, going to a party, or just hanging out, the Blue Elephant Avocado Jacket is the perfect jacket for you.
"""

#### Top k
# - The default value for `top_k` is `40`.
# - You can set `top_k` to values between `1` and `40`.
# - The decoding strategy applies `top_k`, then `top_p`, then `temperature` (in that order).

top_k = 20
top_p = 0.7

response = generation_model.predict(
    prompt=prompt,
    temperature=0.9,
    top_k=top_k,
    top_p=top_p,
)

print(f"[top_p = {top_p}]")
print(response.text)

"""
[top_p = 0.7]
**Introducing the new Blue Elephant Avocado Jacket!**

This jacket is the perfect way to show your love of both blue elephants and avocados. It's made of soft, lightweight fabric that's perfect for any season. The blue elephant print is sure to turn heads, and the avocado-shaped pockets are a fun and functional addition.

Whether you're wearing it to the park, to the beach, or to your next party, the Blue Elephant Avocado Jacket is sure to make you smile. Order yours today!

**Here are some of the benefits of owning a Blue Elephant Avocado Jacket:**
"""
