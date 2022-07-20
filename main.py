from language.parser import GeneticParser, RuleParser
from language.rule import Rule
from viewer.geneticviewer import GeneticViewer
import numpy as np
from language.language import Language
from time import perf_counter
from neural_net.neural_net import NeuralNet

if __name__ == "__main__":
    constraint_types = {"list": lambda x: isinstance(x, list),
                        "nparray": lambda x: isinstance(x, np.ndarray),
                        "float": lambda x: isinstance(x, float)}

    convert_to_nparray = Rule(RuleParser("a", constraints={"a": "list"}),
                              RuleParser("np.array(a)", constraints={"np.array": "nparray"}))

    dot_product = Rule(RuleParser("a", "b", constraints={"a": "nparray", "b": "nparray"}),
                       RuleParser("np.sum(a*b)", constraints={"np.sum": ("nparray", "float")}))

    modules = {"numpy": "np"}

    list1 = [1, 2, 3]
    list2 = [2, 2, 2]

    lang = Language()
    lang.add_rules(convert_to_nparray, dot_product)
    lang.add_types(constraint_types)
    # Or we could say Language(rules=[convert_to_nparray, dot_product], types=constraint_types)

    viewer = GeneticViewer(lang, modules)
    obj = viewer.blob(list1, list2)

    from utilities.util_funcs import test_neural_net
    test_neural_net(lang, device_str="cpu", num_tests=20, num_epochs=100, lr=0.001)
    # output = GeneticParser("a", constraints={"a": "float"})
    # print(obj.view(output))
