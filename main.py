from language.parser import GeneticParser, RuleParser, Node
from language.rule import Rule
from language.rule_search import RuleSearch
from viewer.genetic_viewer import GeneticViewer
from viewer.random_viewer import RandomViewer
import numpy as np
from language.language import Language
from time import perf_counter
from neural_net.neural_net import NeuralNet
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
    list2 = [2, 2, 2]

    lang = Language()
    lang.add_rules(convert_to_nparray, dot_product)
    lang.add_types(constraint_types)
    # Or we could say Language(rules=[convert_to_nparray, dot_product], types=constraint_types)

    viewer = RandomViewer(lang, modules)
    obj = viewer.blob(list1, list2)
    x = NeuralNet(lang)
    # utilities.util_funcs.test_neural_net(lang, n_gen=5, device_str="cpu", num_tests=1, num_epochs=1000)
    
    # viewer = GeneticViewer(lang, modules, n_generations=50, n_gen=8, num_epochs=1000)
    # obj = viewer.blob(list1, list2)
    # output = GeneticParser("a", constraints={"a": "float"})
    # print(obj.view(output))
