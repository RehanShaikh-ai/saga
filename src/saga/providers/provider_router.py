import os
from collections.abc import Generator
from importlib.resources import files
from importlib.resources.abc import Traversable
from typing import Any

import yaml
from dotenv import load_dotenv
from openai import APIError, OpenAI, Stream
from openai.types.chat import ChatCompletionChunk, ChatCompletionMessageParam

if not load_dotenv():
    raise FileNotFoundError(".env file not found.")

configs_path: Traversable = files("saga.configs") / "configs.yaml"

if not configs_path.is_file():
    raise FileNotFoundError(f"Configs couldnt be found at: {configs_path}")

try:
    with configs_path.open("r", encoding="utf-8") as file:
        loaded_config = yaml.safe_load(file)

except yaml.YAMLError as e:
    raise ValueError(f"Error loading configs: {e}") from e

if loaded_config is None:
    raise ValueError("configs.yaml is empty.")

config: dict[str, Any] = loaded_config


class FreeLLMProvider:

    def __init__(self):
        self.api_key: str | None = os.getenv("FREELLMAPI_KEY")
        if self.api_key is None:
            raise ValueError("FREELLMAPI_KEY environment variable is missing")

        self.base_url: str = config["provider"]["freellm_base_url"]

        self.client: OpenAI = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )

    def generate(
        self,
        messages: list[ChatCompletionMessageParam],
        temperature: float = config["provider"]["default_temperature"],
        model: str = config["provider"]["default_model"],
    ) -> str | None:

        try:
            response = self.client.chat.completions.create(
                model=model, messages=messages, temperature=temperature
            )
        except APIError as e:
            raise RuntimeError(f"LLM request failed {e}") from e

        content = response.choices[0].message.content
        if not content:
            return None

        return content

    def list_models(self) -> list[str]:
        models = self.client.models.list()
        if not models.data:
            raise RuntimeError("Failed to fetch model list.")

        return [model.id for model in models.data]

    def stream(
        self,
        messages: list[ChatCompletionMessageParam],
        temperature: float = config["provider"]["default_temperature"],
        model: str = config["provider"]["default_model"],
    ) -> Generator[str, None, None]:

        try:
            stream: Stream[ChatCompletionChunk] = self.client.chat.completions.create(
                model=model, messages=messages, temperature=temperature, stream=True
            )

            for chunk in stream:

                if not chunk.choices:
                    continue

                content: str | None = chunk.choices[0].delta.content

                if content:
                    yield content

        except APIError as e:
            raise RuntimeError(f"LLM request failed {e}") from e


# if __name__ == "__main__":

#     provider = FreeLLMProvider()
#     messages: list[ChatCompletionMessageParam] = [
#         {
#             "role": "user",
#             "content": "say hi",
#         }
#     ]

#     # print(provider.generate(messages=messages, temperature=0.9))

#     print(provider.list_models())

#     # for chunk in provider.stream(messages=messages):
#     #     print(chunk, end="")
#     # print()
