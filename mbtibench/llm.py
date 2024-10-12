import logging
from typing import Dict, List, Optional, Tuple

import tiktoken
from openai import OpenAI
from transformers import AutoTokenizer, PreTrainedTokenizer

from .enums import ModelName

logger = logging.getLogger(__name__)


class LLM:
    def __init__(self, name: ModelName, base_url: str, api_key: str):
        self._model_name = name
        self._model = OpenAI(base_url=base_url, api_key=api_key)
        self._tokenizer = self._get_tokenizer(name)

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        return self._tokenizer

    def _get_tokenizer(self, name: ModelName) -> PreTrainedTokenizer:
        if name.is_gpt4:
            tokenizer = tokenizer = tiktoken.encoding_for_model(name)
        elif name.is_llama3_1:
            tokenizer = AutoTokenizer.from_pretrained("/home/share/models/Meta-Llama-3.1-70B-Instruct")
            tokenizer.add_bos_token = False
        elif name.is_qwen2:
            tokenizer = AutoTokenizer.from_pretrained("/home/share/models/Qwen2-72B-Instruct")
            tokenizer.add_bos_token = False
        else:
            raise ValueError(f"Tokenizer not found for model {name}")

        return tokenizer

    def extract_prompt(self, messages: List[Dict[str, str]]) -> Tuple[Optional[List[Dict[str, str]]], Optional[int]]:
        assert messages[0]["role"] == "system"
        assert len(messages[1:]) % 2 == 0

        for i in range(1, len(messages), 2):
            user_turn, assistant_turn = messages[i], messages[i + 1]
            assert user_turn["role"] == "user"
            assert assistant_turn["role"] == "assistant"

            if "[[PLACEHOLDER]]" in assistant_turn["content"]:
                return messages[: i + 1], i + 1

        return None, None

    def _chat_one_turn(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0,
        max_tokens=2048,
    ) -> List[Dict[str, str]]:
        try:
            logger.info(f"Chatting with {len(messages[1:])} turns")
            response = self._model.chat.completions.create(
                model=str(self._model_name),
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            response_content = response.choices[0].message.content
        except Exception as e:
            response_content = f"OPENAI API ERROR: {e}"

        return response_content

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0,
        max_tokens=2048,
    ) -> List[Dict[str, str]]:
        while True:
            extracted_messages, placeholder_index = self.extract_prompt(messages)
            if extracted_messages is None and placeholder_index is None:
                break
            logger.info(f"Found [[PLACEHOLDER]] in message[{placeholder_index}], chat in new turn")
            response_content = self._chat_one_turn(extracted_messages, temperature, max_tokens)
            messages[placeholder_index]["content"] = response_content

        return messages

    def show_real_prompt(self, messages: List[Dict[str, str]]) -> str:
        return self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
