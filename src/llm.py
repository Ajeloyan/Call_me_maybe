from llm_sdk import Small_LLM_Model
import json
from src.models import PromptItem, FuncItems
import numpy as np
from enum import Enum


class States(Enum):
    EXPECT_OPEN_BRACE = "{"
    EXPECT_NAME_KEY = "name"
    EXPECT_COLON = ":"
    EXPECT_FUNCTION_NAME = "function_name"
    EXPECT_COMMA = ","
    EXPECT_PARAMETER_KEY = "parameters"
    EXPECT_PARAM_VALUE = "param_value"


class ConstrainedDecoder:
    def __init__(self, model: Small_LLM_Model, vocab: dict,
                 functions: list[FuncItems]):
        self.model = model
        self.vocab = vocab
        self.functions = functions

        self.current_state = States.EXPECT_OPEN_BRACE
        self.current_prefix: str = ""
        self.selected_function: FuncItems | None = None

        self.reverse_voc: dict[int, str] = {}
        for token_str, token_id in self.vocab.items():
            self.reverse_voc[token_id] = token_str

    def get_allowed_tokens(self) -> list[int]:
        autorized_tokens: list[int] = []
        if self.current_state == States.EXPECT_OPEN_BRACE:
            for token_str, token_id in self.vocab.items():
                clean_token = token_str.replace("Ġ", "").strip()
                if clean_token == "{":
                    autorized_tokens.append(token_id)

        if self.current_state == States.EXPECT_NAME_KEY:
            for token_str, token_id in self.vocab.items():
                clean_token = token_str.replace("Ġ", "").strip()
                if clean_token == '"name"':
                    autorized_tokens.append(token_id)

        if self.current_state == States.EXPECT_COLON:
            for token_str, token_id in self.vocab.items():
                clean_token = token_str.replace("Ġ", "").strip()
                if clean_token == ":":
                    autorized_tokens.append(token_id)
        
        if self.current_state == States.EXPECT_FUNCTION_NAME:
            if self.current_prefix == "":
                for token_str, token_id in self.vocab.items():
                    clean_token = token_str.replace("Ġ", "").strip()
                    if clean_token == '"':
                        autorized_tokens.append(token_id)
            else:
                for token_str, token_id in self.vocab.items():
                    clean_token = token_str.replace("Ġ", "").strip()

                    word_test = self.current_prefix + clean_token
                    is_valid_token = False

                    for func in self.functions:
                        target = f'"{func.name}"'

                        if target.startswith(word_test):
                            is_valid_token = True
                            break
                    if is_valid_token:
                        autorized_tokens.append(token_id)

        if self.current_state == States.EXPECT_COMMA:
            for token_str, token_id in self.vocab.items():
                clean_token = token_str.replace("Ġ", "").strip()
                if clean_token == ",":
                    autorized_tokens.append(token_id)

        if self.current_state == States.EXPECT_PARAMETER_KEY:
            for token_str, token_id in self.vocab.items():
                clean_token = token_str.replace("Ġ", "").strip()
                if clean_token == '"parameters"':
                    autorized_tokens.append(token_id)

        return autorized_tokens

    def consume_token(self, chosen_token_id: int) -> None:
        pass

    def apply_mask(self, logits: np.array,
                   allowed_tokens: list[int]) -> np.ndarray:
        pass

    def generate_function_call(self, prompt: PromptItem) -> dict:
        pass


# def llm_test(prompts: list[PromptItem], func: list[FuncItems]) -> None:
#     model = Small_LLM_Model()
#     vocab_path = model.get_path_to_vocab_file()
#     with open(vocab_path, "r") as file:
#         voc = json.load(file)

#     first_prompt = "can you"
#     ids = model.encode(first_prompt)
#     input_ids = ids.tolist()[0]

#     for _ in range(100):
#         logits = model.get_logits_from_input_ids(input_ids)
#         logits_array = np.array(logits)

#         # id_brace = voc.get("{")
#         # logits_array[id_brace] = 0.0

#         # logits_array[:] = -np.inf

#         best_token_id = int(np.argmax(logits_array))
#         input_ids.append(best_token_id)
#         # for token, token_id in voc.items():
#         #     if token_id == best_token_id:
#         #         first_prompt += token
#     phrase_finale = model.decode(input_ids)
#     print(phrase_finale)
