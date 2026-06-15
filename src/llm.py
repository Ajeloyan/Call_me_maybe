from llm_sdk import Small_LLM_Model
import json
from src.models import PromptItem, FuncItems
import numpy as np
from enum import Enum


class States(Enum):
    EXPECT_OPEN_BRACE = "{"
    EXPECT_NAME_KEY = "name"
    EXPECT_NAME_COLON = ":"
    EXPECT_FUNCTION_NAME = "function_name"
    EXPECT_NAME_COMMA = ","
    EXPECT_PARAMETER_KEY = "parameters"
    EXPECT_COLON_PARAM = ":"
    EXPECT_PARAM_BRACE_O = "{"
    EXPECT_ARGUMENT_NAME = "param_name"
    EXPECT_COLON_PARAM_IN = ":"
    EXPECT_PARAM_VALUE = "param_value"
    EXPECT_PARAM_COMMA = ","
    EXPECT_PARAM_BRACE_C = "}"
    EXPECT_CLOSE_BRACE = "}"


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

        self.generated_params: list[str] = []
        self.current_param_name: str | None = None

    def get_allowed_tokens(self) -> list[int]:
        autorized_tokens: list[int] = []
        if self.current_state == States.EXPECT_OPEN_BRACE:
            for token_str, token_id in self.vocab.items():
                clean_token = token_str.replace("Ġ", "").strip()
                if clean_token == "{":
                    autorized_tokens.append(token_id)

        if self.current_state == States.EXPECT_NAME_KEY:
            target = '"name"'
            for token_str, token_id in self.vocab.items():
                clean_token = token_str.replace("Ġ", "").strip()
                if clean_token == "":
                    continue
                word_test = self.current_prefix + clean_token

                if target.startswith(word_test):
                    autorized_tokens.append(token_id)

        if self.current_state == States.EXPECT_NAME_COLON:
            for token_str, token_id in self.vocab.items():
                clean_token = token_str.replace("Ġ", "").strip()
                if clean_token == ":":
                    autorized_tokens.append(token_id)

        if self.current_state == States.EXPECT_FUNCTION_NAME:
            if self.current_prefix == "":
                for token_str, token_id in self.vocab.items():
                    clean_token = token_str.replace("Ġ", "").strip()

                    if clean_token == "":
                        continue

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

        if self.current_state == States.EXPECT_NAME_COMMA:
            for token_str, token_id in self.vocab.items():
                clean_token = token_str.replace("Ġ", "").strip()
                if clean_token == ",":
                    autorized_tokens.append(token_id)

        if self.current_state == States.EXPECT_PARAMETER_KEY:
            target = '"parameters"'
            for token_str, token_id in self.vocab.items():
                clean_token = token_str.replace("Ġ", "").strip()
                if clean_token == "":
                    continue
                word_test = self.current_prefix + clean_token
                if target.startswith(word_test):
                    autorized_tokens.append(token_id)

        if self.current_state == States.EXPECT_COLON_PARAM:
            for token_str, token_id in self.vocab.items():
                clean_token = token_str.replace("Ġ", "").strip()
                if clean_token == ":":
                    autorized_tokens.append(token_id)

        if self.current_state == States.EXPECT_PARAM_BRACE_O:
            for token_str, token_id in self.vocab.items():
                clean_token = token_str.replace("Ġ", "").strip()
                if clean_token == "{":
                    autorized_tokens.append(token_id)

        if self.current_state == States.EXPECT_ARGUMENT_NAME:
            if self.current_prefix == "":
                for token_str, token_id in self.vocab.items():
                    clean_token = token_str.replace("Ġ", "").strip()
                    if clean_token == "":
                        continue
                    if clean_token == '"':
                        autorized_tokens.append(token_id)
            else:
                for token_str, token_id in self.vocab.items():
                    clean_token = token_str.replace("Ġ", "").strip()

                    word_test = self.current_prefix + clean_token
                    is_valid_token = False

                    for param in self.selected_function.parameters:
                        if param in self.generated_params:
                            continue

                        target = f'"{param}"'

                        if target.startswith(word_test):
                            is_valid_token = True
                            break
                    if is_valid_token:
                        autorized_tokens.append(token_id)

        if self.current_state == States.EXPECT_COLON_PARAM_IN:
            for token_str, token_id in self.vocab.items():
                clean_token = token_str.replace("Ġ", "").strip()
                if clean_token == ":":
                    autorized_tokens.append(token_id)

        if self.current_state == States.EXPECT_PARAM_VALUE:
            expected_type = self.selected_function.parameters[
                self.current_param_name].type
            if expected_type in ["int", "number"]:
                is_last_param = len(self.selected_function.parameters) == len(
                    self.generated_params)
                for token_str, token_id in self.vocab.items():
                    clean_token = token_str.replace("Ġ", "").strip()
                    if clean_token and all(char.isdigit() or char in "-." for char in clean_token):
                        autorized_tokens.append(token_id)
                    elif clean_token == "," and not is_last_param:
                        autorized_tokens.append(token_id)
                    elif clean_token == "}" and is_last_param:
                        autorized_tokens.append(token_id)
            if expected_type == "string":
                if self.current_prefix == "":
                    for token_str, token_id in self.vocab.items():
                        clean_token = token_str.replace("Ġ", "").strip()
                        if clean_token == '"':
                            autorized_tokens.append(token_id)
                else:
                    for token_str, token_id in self.vocab.items():
                        autorized_tokens.append(token_id)

            if expected_type == "boolean":
                for token_str, token_id in self.vocab.items():
                    clean_token = token_str.replace("Ġ", "").strip()
                    if clean_token == "false" or clean_token == "true":
                        autorized_tokens.append(token_id)

        if self.current_state == States.EXPECT_PARAM_COMMA:
            for token_str, token_id in self.vocab.items():
                clean_token = token_str.replace("Ġ", "").strip()
                if clean_token == ",":
                    autorized_tokens.append(token_id)

        if self.current_state == States.EXPECT_PARAM_BRACE_C:
            for token_str, token_id in self.vocab.items():
                clean_token = token_str.replace("Ġ", "").strip()
                if clean_token == "}":
                    autorized_tokens.append(token_id)

        if self.current_state == States.EXPECT_CLOSE_BRACE:
            for token_str, token_id in self.vocab.items():
                clean_token = token_str.replace("Ġ", "").strip()
                if clean_token == "}":
                    autorized_tokens.append(token_id)

        return autorized_tokens

    def consume_token(self, chosen_token_id: int) -> None:
        if self.current_state == States.EXPECT_OPEN_BRACE:
            self.current_state = States.EXPECT_NAME_KEY
            return

        if self.current_state == States.EXPECT_NAME_KEY:
            token_str = self.reverse_voc[chosen_token_id]
            clean_token = token_str.replace("Ġ", "").strip()
            self.current_prefix += clean_token

            if self.current_prefix == '"name"':
                self.current_state = States.EXPECT_NAME_COLON
                self.current_prefix = ""
            return

        if self.current_state == States.EXPECT_NAME_COLON:
            self.current_state = States.EXPECT_FUNCTION_NAME
            return

        if self.current_state == States.EXPECT_FUNCTION_NAME:
            token_str = self.reverse_voc[chosen_token_id]
            clean_token = token_str.replace("Ġ", "").strip()
            self.current_prefix += clean_token
            for func in self.functions:
                if self.current_prefix == f'"{func.name}"':
                    self.selected_function = func
                    self.current_prefix = ""
                    self.current_state = States.EXPECT_NAME_COMMA
                    return
            return

        if self.current_state == States.EXPECT_NAME_COMMA:
            self.current_state = States.EXPECT_PARAMETER_KEY
            return

        if self.current_state == States.EXPECT_PARAMETER_KEY:
            token_str = self.reverse_voc[chosen_token_id]
            clean_token = token_str.replace("Ġ", "").strip()
            self.current_prefix += clean_token

            if self.current_prefix == '"parameters"':
                self.current_state = States.EXPECT_COLON_PARAM
                self.current_prefix = ""
            return

        if self.current_state == States.EXPECT_COLON_PARAM:
            self.current_state = States.EXPECT_PARAM_BRACE_O
            return

        if self.current_state == States.EXPECT_PARAM_BRACE_O:
            self.current_state = States.EXPECT_ARGUMENT_NAME
            return

        if self.current_state == States.EXPECT_ARGUMENT_NAME:
            token_str = self.reverse_voc[chosen_token_id]
            clean_token = token_str.replace("Ġ", "").strip()
            self.current_prefix += clean_token

            for param in self.selected_function.parameters:
                if self.current_prefix == f'"{param}"':
                    self.generated_params.append(param)
                    self.current_param_name = param
                    self.current_prefix = ""
                    self.current_state = States.EXPECT_COLON_PARAM_IN
                    return
            return

        if self.current_state == States.EXPECT_COLON_PARAM_IN:
            self.current_state = States.EXPECT_PARAM_VALUE
            return

        if self.current_state == States.EXPECT_PARAM_VALUE:
            expected_type = self.selected_function.parameters[self.current_param_name].type

            token_str = self.reverse_voc[chosen_token_id]
            clean_token = token_str.replace("Ġ", "").strip()

            if expected_type == "string":

                if self.current_prefix == "":
                    self.current_prefix += clean_token
                    return

                else:
                    self.current_prefix += clean_token

                    if '"' in clean_token:
                        if len(self.selected_function.parameters) == len(self.generated_params):
                            self.current_state = States.EXPECT_PARAM_BRACE_C
                        else:
                            self.current_state = States.EXPECT_PARAM_COMMA
                        self.current_prefix = ""
                        return
                    return
            elif expected_type in ["int", "number"]:
                self.current_prefix += clean_token

                if "," in clean_token:
                    self.current_state = States.EXPECT_ARGUMENT_NAME
                    self.current_prefix = ""
                    return
                elif "}" in clean_token:
                    self.current_state = States.EXPECT_CLOSE_BRACE
                    self.current_prefix = ""
                    return
                return
            
            elif expected_type == "boolean":
                if len(self.selected_function.parameters) == len(self.generated_params):
                    self.current_state = States.EXPECT_PARAM_BRACE_C
                else:
                    self.current_state = States.EXPECT_PARAM_COMMA
                return

        if self.current_state == States.EXPECT_PARAM_COMMA:
            self.current_state = States.EXPECT_ARGUMENT_NAME
            return

        if self.current_state == States.EXPECT_PARAM_BRACE_C:
            self.current_state = States.EXPECT_CLOSE_BRACE
            return

    def apply_mask(self, logits: np.array,
                   allowed_tokens: list[int]) -> np.ndarray:
        mask = np.full_like(logits, -np.inf, dtype=float)
        mask[allowed_tokens] = logits[allowed_tokens]
        return mask

    def generate_function_call(self, prompt: PromptItem) -> dict:
        ids = self.model.encode(prompt.prompt)
        input_ids = ids.tolist()[0]

        generated_ids = []

        for _ in range(200):
            logits = self.model.get_logits_from_input_ids(input_ids)
            logits_array = np.array(logits)

            allowed_tokens = self.get_allowed_tokens()

            masked_logits = self.apply_mask(logits_array, allowed_tokens)

            best_token_id = int(np.argmax(masked_logits))

            input_ids.append(best_token_id)
            generated_ids.append(best_token_id)

            self.consume_token(best_token_id)

            if self.current_state == States.EXPECT_CLOSE_BRACE:
                break

        json_str = self.model.decode(generated_ids)

        print("\n=== SCÈNE DE CRIME ===")
        print(f"Texte généré : {repr(json_str)}")
        print("======================\n")

        return json.loads(json_str)
