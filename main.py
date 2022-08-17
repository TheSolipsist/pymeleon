from language.parser import GeneticParser, RuleParser
from language.rule import Rule
from viewer.genetic_viewer import GeneticViewer
import numpy as np
from language.language import Language
import utilities.util_funcs

if __name__ == "__main__":
    constraint_types = {"list": lambda x: isinstance(x, list),
                        "nparray": lambda x: isinstance(x, np.ndarray),
                        "float": lambda x: isinstance(x, float)}

    convert_to_nparray = Rule(RuleParser("a", constraints={"a": "list"}),
                              RuleParser("np.array(a)", constraints={"np.array": "nparray"}))

    dot_product = Rule(RuleParser("a", "b", constraints={"a": "nparray", "b": "nparray"}),
                       RuleParser("np.sum(a*b)", constraints={"np.sum": "float"}))

    modules = {"numpy": "np"}

    list1 = [1, 2, 3]
    list2 = [2, 2, 4]

    lang = Language()
    lang.add_rules(convert_to_nparray, dot_product)
    lang.add_types(constraint_types)
    # Or we could say Language(rules=[convert_to_nparray, dot_product], types=constraint_types)

    # utilities.util_funcs.test_neural_net(lang, {"n_gen": 20, "num_epochs": 300, "lr": 0.001}, device_str="cuda", num_tests=2)
    
    viewer = GeneticViewer(lang, modules, n_iter=10, n_fittest=5, n_gen=10, 
                           fitness="neural_random", hyperparams={"n_gen": 20, "num_epochs": 300, "lr": 0.001}, device_str="cuda")
    obj = viewer.blob(list1, list2)
    output = GeneticParser("a", constraints={"a": "float"})
    print(obj.view(output))
