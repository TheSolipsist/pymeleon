from language.parser import GeneticParser, RuleParser
from language.rule import Rule
from viewer.geneticviewer import GeneticViewer
import numpy as np
from language.language import Language
from neural_net.training_generation import dfs_representation
from time import perf_counter

constraint_types = {"islist": lambda x: isinstance(x, list),
                    "isnparray": lambda x: isinstance(x, np.ndarray),
                    "isfloat": lambda x: isinstance(x, float)}

convert_to_nparr = Rule(RuleParser("a", constraints={"a": "islist"}),
                        RuleParser("np.array(a)", constraints={"np.array": "isnparray"}))

dot_product = Rule(RuleParser("a", "b", constraints={"a": "isnparray", "b": "isnparray"}),
                   RuleParser("np.sum(a*b)", constraints={"np.sum": ("isnparray", "isfloat")}))

modules = {"numpy": "np"}
    
list1 = [1, 2, 3]
list2 = [2, 2, 2]

lang = Language()
lang.add_rules(convert_to_nparr, dot_product)
lang.add_types(constraint_types)
# Or we could say Language(rules=[convert_to_nparr, dot_product], types=constraint_types)

viewer = GeneticViewer(lang, modules)
obj = viewer.blob(list1, list2)
print(dfs_representation(obj.get_graph(), lang))
# output = GeneticParser("a", constraints={"a": "isfloat"})
# print(obj.view(output))
