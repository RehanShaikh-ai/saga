import os
from collections.abc import Generator

import yaml
from dotenv import load_dotenv
from openai import OpenAI, Stream
from openai.types.chat import ChatCompletionChunk, ChatCompletionMessageParam

load_dotenv()
configs_file_path = "src/saga/configs/configs.yaml"

if not os.path.isfile(configs_file_path):
    raise FileNotFoundError(f"Configs couldnt be found at: {configs_file_path}")

try:
    with open("src/saga/configs/configs.yaml", "r") as file:
        config = yaml.safe_load(file)

except yaml.YAMLError as e:
    raise yaml.YAMLError(f"Error loading configs: {e}")


class FreeLLMProvider:

    def __init__(self):

        self.api_key: str | None = os.getenv("FREELLMAPI_KEY")

        if self.api_key is None:
            raise ValueError("FREELLMAPI_KEY environment variable is missing")

        self.client: OpenAI = OpenAI(
            base_url=config["PROVIDER"]["FREELLM_BASE_URL"],
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

        if not response.choices[0].message.content:
            return None

        return response.choices[0].message.content

    def list_models(self) -> list[str]:
        models = self.client.models.list()
        return [model.id for model in models.data]

    def stream(
        self,
        messages: list[ChatCompletionMessageParam],
        temperature: float = 0.5,
        model: str = "bazaarlink-auto",
    ) -> Generator[str, None, None]:

        stream: Stream[ChatCompletionChunk] = self.client.chat.completions.create(
            model=model, messages=messages, temperature=temperature, stream=True
        )

        for chunk in stream:

            if not chunk.choices:
                continue

            content: str | None = chunk.choices[0].delta.content

            if content:
                yield content


if __name__ == "__main__":

    provider = FreeLLMProvider()
    messages: list[ChatCompletionMessageParam] = [
        {
            "role": "user",
            "content": "say hi",
        }
    ]

    print(provider.generate(messages=messages, temperature=0.9))

    print(provider.list_models())

    for chunk in provider.stream(messages=messages):
        print(chunk, end="")
    print()
