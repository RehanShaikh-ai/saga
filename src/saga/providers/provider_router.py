import os

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam


class FreeLLMProvider:

    def __init__(self):
        load_dotenv()
        self.api_key: str | None = os.getenv("FREELLMAPI_KEY")

        if self.api_key is None:
            raise ValueError("FREELLMAPI_KEY environment variable is missing")

        self.client: OpenAI = OpenAI(
            base_url="http://localhost:5173/v1",
            api_key=self.api_key,
        )

    def generate(
        self,
        messages: list[ChatCompletionMessageParam],
        temperature: float = 0.5,
        model: str = "bazaarlink-auto",
    ) -> str | None:

        response = self.client.chat.completions.create(
            model=model, messages=messages, temperature=temperature
        )

        return response.choices[0].message.content

    def list_models(self) -> list[str]:
        models = self.client.models.list()
        return [model.id for model in models.data]


provider = FreeLLMProvider()
messages: list[ChatCompletionMessageParam] = [
    {
        "role": "user",
        "content": "Hello! Introduce yourself in one sentence.",
    }
]
print(provider.generate(messages=messages, temperature=0.9))

print(provider.list_models())
