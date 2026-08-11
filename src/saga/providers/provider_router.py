import os

from dotenv import load_dotenv
from openai import OpenAI


class FreeLLMProvider:

    def __init__(self):
        load_dotenv()
        self.API_KEY = os.getenv("FREELLMAPI_KEY")
        self.client = OpenAI(
            base_url="http://localhost:5173/v1",
            api_key=self.API_KEY,
        )

    def generate(self, messages, temperature=0.5, model="bazaarlink-auto"):

        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature
        )

        return response.choices[0].message.content


    def list_models(self):
        models = self.client.models.list()
        return [model.id for model in models.data]

provider = FreeLLMProvider()
messages = [
    {
        "role": "user",
        "content": "Hello! Introduce yourself in one sentence.",
    }
]
print(provider.generate(messages=messages, temperature=0.9))

print(provider.list_models())


