from .models import PromptItem, FuncItems
from pydantic import ValidationError
import json
import sys


def parse_prompts(filepath: str) -> list:
    try:
        with open(filepath, "r") as file:
            data = json.load(file)
            prompts_list = []
            for item in data:
                prompt = PromptItem(**item)
                prompts_list.append(prompt)
            return prompts_list
    except (ValidationError, FileNotFoundError,
            PermissionError, json.JSONDecodeError) as e:
        print(e)
        sys.exit(1)


def parse_functions(filepath: str) -> list:
    try:
        with open(filepath, "r") as file:
            data = json.load(file)
            func_list = []
            for item in data:
                func = FuncItems(**item)
                func_list.append(func)
            return func_list
    except (ValidationError, FileNotFoundError,
            PermissionError, json.JSONDecodeError) as e:
        print(e)
        sys.exit(1)
