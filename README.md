# 🧠✨ My Generative AI Journey with Google Cloud: Building Real-World Apps

I've always been curious about how the AI we chat with — like assistants and copilots — actually works under the hood. While my first badge in **Prompt Design** gave me the vocabulary to "speak AI," my second badge took things a level deeper. This time, I wasn’t just prompting a model — I was building with it.

The **[Google Cloud Gen AI Exchange Program](https://vision.hack2skill.com/event/genaiexchange?utm_source=hack2skill&utm_medium=homepage)** introduced me to powerful tools like **Gemini** (for conversation-based AI) and **Imagen** (for image generation), and showed me how real-world AI applications are built — from architecture and APIs to stream handling and safety measures.

It was like going from learning a language to writing my first screenplay with it.

---

## 🎯 1. Build an AI Image Recognition App using Gemini on Vertex AI

The journey began with teaching AI how to see. In this module, I used **Google’s Gemini model** to create an app that recognizes and describes images. With just a few lines of code, I built a system that could tell whether I uploaded a forest, a cat, or a flower bouquet — and describe it beautifully.

✨ **Takeaway**: AI isn’t just about data; it’s about perception.

---

## 🖼️ 2. Build an AI Image Generator App using Imagen on Vertex AI

Next, it was time to flip the script — from understanding images to generating them. Using the **Imagen model**, I created visually rich artwork from simple prompts like:

> *"A sunset over a lavender field with flying dandelions"*

I built an app that could generate images on demand — no Photoshop required.

🎨 **Takeaway**: With the right prompt and a bit of code, I could turn imagination into images.

---

## 🔧 Getting Practical with Gemini: Chat Without Stream

My first experiment involved creating a conversational agent using **Gemini 2.0 Flash**. The code below shows how I maintained context across interactions:

```python
from google import genai
from google.genai.types import HttpOptions, ModelContent, Part, UserContent

import logging
from google.cloud import logging as gcp_logging

# ------ Qwiklabs internal logging setup --------
gcp_logging_client = gcp_logging.Client()
gcp_logging_client.setup_logging()

client = genai.Client(
    vertexai=True,
    project="your-project-id",
    location="europe-west1",
    http_options=HttpOptions(api_version="v1")
)

chat = client.chats.create(
    model="gemini-2.0-flash-001",
    history=[
        UserContent(parts=[Part(text="Hello")]),
        ModelContent(parts=[Part(text="Great to meet you. What would you like to know?")]),
    ],
)

response = chat.send_message("What are all the colors in a rainbow?")
print(response.text)

response = chat.send_message("Why does it appear when it rains?")
print(response.text)
```

🌈 The model maintained context beautifully — like a natural conversation.

---

## 🔄 Now with Stream: Making AI Conversations Feel Real-Time

Next, I implemented **streaming responses**, where the model outputs content word-by-word. Perfect for responsive user experiences:

```python
from google import genai
from google.genai.types import HttpOptions

import logging
from google.cloud import logging as gcp_logging

# ------ Qwiklabs internal logging setup --------
gcp_logging_client = gcp_logging.Client()
gcp_logging_client.setup_logging()

client = genai.Client(
    vertexai=True,
    project="your-project-id",
    location="europe-west1",
    http_options=HttpOptions(api_version="v1")
)

chat = client.chats.create(model="gemini-2.0-flash-001")
response_text = ""

for chunk in chat.send_message_stream("What are all the colors in a rainbow?"):
    print(chunk.text, end="")
    response_text += chunk.text
```

🧠 **Takeaway**: Streaming responses make AI interactions feel alive and fluid.

---

## 🧠✨ 4. Build a Multi-Modal GenAI Application: Challenge Lab

The real challenge: Combine image generation and text generation into a single app.

### ✅ Task 1: Create a Bouquet with Imagen

Using the `imagen-3.0-generate-002` model to generate a bouquet image:

```python
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel

def generate_image(project_id, location, output_file, prompt):
    vertexai.init(project=project_id, location=location)
    model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-002")
    images = model.generate_images(
        prompt=prompt,
        number_of_images=1,
        seed=1,
        add_watermark=False,
    )
    images[0].save(location=output_file)
    print(f"Image saved to {output_file}")
    return images

# Generate the bouquet
generate_image(
    project_id="your-project-id",
    location="europe_west1",
    output_file="bouquet.png",
    prompt="Create an image containing a bouquet of 2 sunflowers and 3 roses"
)
```

### ✅ Task 2: Analyze the Bouquet & Generate a Birthday Wish with Gemini

Using the image as input, Gemini generated a poetic birthday wish:

```python
import vertexai 
from vertexai.generative_models import GenerativeModel, Part, Image

def analyze_bouquet_image(image_path):
    vertexai.init(
        project='your-project-id',
        location='europe-west1',
    )

    model = GenerativeModel("gemini-2.0-flash-001")
    image_input = Part.from_image(Image.load_from_file(location=image_path))

    messages = [
        "Generate a birthday wish based on the following image",
        image_input
    ]

    chat = model.start_chat()
    response = chat.send_message(content=messages, stream=False)

    result_text = response.text
    print(result_text)

    with open("analysis_log.txt", "w") as log_file:
        log_file.write(result_text)

    print("✅ Log file created: analysis_log.txt")

# Run the function
analyze_bouquet_image("bouquet.png")
```

🎁 **Takeaway**: When AI understands visuals and responds creatively, it becomes more than a tool — it becomes a collaborator.

---

## 🔚 Final Thoughts

By the end of this series, I wasn’t just building with GenAI — I was collaborating with it. From meaningful visuals to heartfelt words, the **multi-modal power** of Vertex AI opened up new ways to express, create, and connect.

If you’re curious about the **[Google Cloud Gen AI Exchange Program](https://vision.hack2skill.com/event/genaiexchange?utm_source=hack2skill&utm_medium=homepage)** and want to dive in, feel free to reach out!

👉 **[Let’s connect on LinkedIn](https://www.linkedin.com/in/sanjana-nathani-26a42727b/)**

If you missed my earlier write-up on Prompt Design, check it out here:
🔗 **[Building Strong Foundations with Prompt Design on Vertex AI](https://github.com/Sanjana006/GenAI-Academy-Badge-1/blob/main/README.md)**

---

## 🙏 Special Thanks

- **[Google Cloud](https://www.cloudskillsboost.google)**  
- **[Hack2skill](https://vision.hack2skill.com/event/genaiexchange?utm_source=hack2skill&utm_medium=homepage)**  

---

## 🏷️ Tags

`#GenAIExchange` `#GenAIAcademy` `#VertexAI` `#Gemini` `#Imagen` `#MultiModalAI` `#GoogleCloud` `#AIApps` `#PromptEngineering` `#LifelongLearning`

> *Together, let’s build the future of AI — one prompt, one pixel, and one conversation at a time.*
