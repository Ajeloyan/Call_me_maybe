from llm_sdk import Small_LLM_Model
import json
from src.models import PromptItem, FuncItems
import numpy as np
from enum import Enum


class States(Enum):
    pass


def llm_test(prompts: list[PromptItem], func: list[FuncItems]) -> None:
    model = Small_LLM_Model()
    vocab_path = model.get_path_to_vocab_file()
    with open(vocab_path, "r") as file:
        voc = json.load(file)

    first_prompt = "can you"
    ids = model.encode(first_prompt)
    input_ids = ids.tolist()[0]

    for _ in range(100):
        logits = model.get_logits_from_input_ids(input_ids)
        logits_array = np.array(logits)

        # id_brace = voc.get("{")
        # logits_array[id_brace] = 0.0

        # logits_array[:] = -np.inf

        best_token_id = int(np.argmax(logits_array))
        input_ids.append(best_token_id)
        # for token, token_id in voc.items():
        #     if token_id == best_token_id:
        #         first_prompt += token
    phrase_finale = model.decode(input_ids)
    print(phrase_finale)
