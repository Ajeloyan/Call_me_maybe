from .parser import parse_prompts, parse_functions


if __name__ == "__main__":
    filepath_p = "data/input/function_calling_tests.json"
    filepath_f = "data/input/functions_definition.json"

    prompts_list = parse_prompts(filepath_p)
    functions = parse_functions(filepath_f)

    print("Prompts chargés :", len(prompts_list) if prompts_list else 0)
    print("Fonctions chargées :", len(functions) if functions else 0)

