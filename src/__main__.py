from .parser import parse_prompts, parse_functions
from .llm import run_pipeline
from llm_sdk import Small_LLM_Model
import json

if __name__ == "__main__":
    filepath_p = "data/input/function_calling_tests.json"
    filepath_f = "data/input/functions_definition.json"
    output_path = "data/output/function_calling_results.json"

    prompts_list = parse_prompts(filepath_p)
    functions = parse_functions(filepath_f)

    model = Small_LLM_Model()
    vocab_path = model.get_path_to_vocab_file()

    with open(vocab_path, "r") as file:
        vocab = json.load(file)

    results = run_pipeline(prompts_list, functions, model, vocab, output_path)
    print(json.dumps(results, indent=4, ensure_ascii=False))
